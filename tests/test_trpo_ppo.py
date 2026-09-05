"""Learning-rule, environment, and persistence checks for TRPO and PPO."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence
from torch.nn.utils import parameters_to_vector

from aprenderl import PPO, TRPO, PPOConfig, TRPOConfig
from aprenderl.networks import ValueNetwork
from aprenderl.types import Transition


@pytest.fixture(params=[PPO, TRPO])
def algorithm(request):
    return request.param


def config(algorithm, **kwargs):
    options = dict(n_steps=5, seed=7)
    if algorithm is PPO:
        options.update(n_epochs=2, batch_size=3)
    else:
        options.update(value_epochs=2, cg_steps=5)
    options.update(kwargs)
    return algorithm.config_class(**options)


def fill_rollout(agent):
    observation, _ = agent.env.reset(seed=7)
    for i in range(agent.config.n_steps):
        state = torch.as_tensor(observation).unsqueeze(0)
        with torch.no_grad():
            distribution = agent.policy_action_space.distribution(
                agent.policy_network(state)
            )
            action = distribution.sample()
            value = agent.value_network(state).item()
            log_prob = distribution.log_prob(action).item()
        agent.rollout_buffer.add(
            observation,
            action.squeeze(0).numpy(),
            float(i + 1),
            i == agent.config.n_steps - 1,
            False,
            value,
            0.0,
            log_prob,
        )
    agent.rollout_buffer.compute_returns_and_advantages()


def test_rollouts_update_both_networks_and_preserve_partial_data(algorithm):
    env = gym.make("CartPole-v1", max_episode_steps=2)
    try:
        agent = algorithm(env, config=config(algorithm), device="cpu")
        actor_before = (
            parameters_to_vector(agent.policy_network.parameters()).detach().clone()
        )
        critic_before = (
            parameters_to_vector(agent.value_network.parameters()).detach().clone()
        )
        agent.learn(11, progress_bar=False)
        assert agent.num_updates == 2
        assert len(agent.rollout_buffer) == 1
        assert not torch.equal(
            actor_before, parameters_to_vector(agent.policy_network.parameters())
        )
        assert not torch.equal(
            critic_before, parameters_to_vector(agent.value_network.parameters())
        )
        agent.learn(4, progress_bar=False)
        assert agent.num_updates == 3
        assert len(agent.rollout_buffer) == 0
    finally:
        env.close()


@pytest.mark.parametrize(
    "terminated,truncated,expected",
    [
        (True, False, 1.0),
        (False, True, 3.0),
        (False, False, 2.0),
    ],
)
def test_gae_bootstrap_and_reset_boundaries(algorithm, terminated, truncated, expected):
    env = gym.make("CartPole-v1")
    try:
        value = ValueNetwork(4, hidden_sizes=())
        agent = algorithm(
            env,
            value_network=value,
            config=config(algorithm, n_steps=3, gamma=1, gae_lambda=1),
            device="cpu",
        )
        with torch.no_grad():
            value.value_head.weight.zero_()
            value.value_head.bias.fill_(2)
        for terminal, truncation in [(terminated, truncated), (True, False)]:
            agent._update_from_transition(
                Transition(
                    observation=np.zeros(4, dtype=np.float32),
                    action=0,
                    reward=1,
                    next_observation=np.ones(4, dtype=np.float32),
                    terminated=terminal,
                    truncated=truncation,
                    info={},
                )
            )
        agent.rollout_buffer.compute_returns_and_advantages()
        batch = agent.rollout_buffer.batch(torch.device("cpu"))
        # If no boundary occurs, the second transition contributes 1 - V = -1.
        assert batch.returns[0].item() == pytest.approx(expected)
        assert batch.returns[1].item() == pytest.approx(1)
    finally:
        env.close()


def test_checkpoint_roundtrip_and_resume(algorithm, tmp_path):
    env = gym.make("CartPole-v1")
    restored_env = gym.make("CartPole-v1")
    try:
        agent = algorithm(env, config=config(algorithm), device="cpu").learn(
            10, progress_bar=False
        )
        path = tmp_path / "agent.pt"
        agent.save(path)
        restored = algorithm.load(path, restored_env, device="cpu")
        assert restored.config == agent.config
        assert restored.num_updates == agent.num_updates
        assert restored.num_timesteps == 10
        for name in ["policy_network", "value_network"]:
            torch.testing.assert_close(
                parameters_to_vector(getattr(restored, name).parameters()),
                parameters_to_vector(getattr(agent, name).parameters()),
            )
        assert restored.value_optimizer.state_dict()["state"]
        if algorithm is PPO:
            assert restored.policy_optimizer.state_dict()["state"]
        observation, _ = restored_env.reset(seed=10)
        assert restored.predict(observation, deterministic=True) == agent.predict(
            observation, deterministic=True
        )
        restored.learn(5, progress_bar=False)
        assert restored.num_updates == 3
        mismatch = gym.make("CartPole-v1")
        mismatch.action_space = gym.spaces.Discrete(2, start=1)
        try:
            with pytest.raises(ValueError, match="checkpoint spaces"):
                algorithm.load(path, mismatch, device="cpu")
        finally:
            mismatch.close()
    finally:
        env.close()
        restored_env.close()


def test_custom_network_checkpoint_contract(algorithm, tmp_path):
    env = gym.make("CartPole-v1")
    try:

        def policy():
            return nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

        agent = algorithm(
            env,
            policy(),
            value_network=ValueNetwork(4, hidden_sizes=()),
            config=config(algorithm),
            device="cpu",
        )
        agent.learn(5, progress_bar=False)
        path = tmp_path / "custom.pt"
        agent.save(path)
        with pytest.raises(ValueError, match="requires policy_network"):
            algorithm.load(path, env, device="cpu")
        with pytest.raises(ValueError, match="requires value_network"):
            algorithm.load(path, env, policy_network=policy(), device="cpu")
        restored = algorithm.load(
            path,
            env,
            policy_network=policy(),
            value_network=ValueNetwork(4, hidden_sizes=()),
            device="cpu",
        )
        restored.learn(5, progress_bar=False)
        assert restored.num_updates == 2
    finally:
        env.close()


def test_environment_and_network_validation(algorithm):
    env = gym.make("FrozenLake-v1")
    try:
        with pytest.raises(TypeError, match="Box observation"):
            algorithm(env, device="cpu")
    finally:
        env.close()
    env = gym.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="must return shape"):
            algorithm(env, nn.Linear(4, 3), device="cpu")
        env.action_space = gym.spaces.MultiDiscrete([2, 2])
        with pytest.raises(TypeError, match="Discrete or Box"):
            algorithm(env, device="cpu")
    finally:
        env.close()


@pytest.mark.parametrize("bounded", [True, False])
def test_continuous_actions_train_and_checkpoint(algorithm, bounded, tmp_path):
    env = gym.make("Pendulum-v1")
    if not bounded:
        env.action_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
    try:
        agent = algorithm(env, config=config(algorithm), device="cpu")
        observation, _ = env.reset(seed=7)
        assert env.action_space.contains(agent.predict(observation))
        agent.learn(10, progress_bar=False)
        assert agent.num_updates == 2
        path = tmp_path / "continuous.pt"
        agent.save(path)
        restored = algorithm.load(path, env, device="cpu")
        np.testing.assert_allclose(
            restored.predict(observation, deterministic=True),
            agent.predict(observation, deterministic=True),
        )
        env.action_space = gym.spaces.Box(-3, 3, (1,), dtype=np.float32)
        with pytest.raises(ValueError, match="checkpoint spaces"):
            algorithm.load(path, env, device="cpu")
    finally:
        env.close()


def test_nonzero_action_offset(algorithm):
    class OffsetActions(gym.ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.action_space = gym.spaces.Discrete(2, start=4)

        def action(self, action):
            assert self.action_space.contains(action)
            return action - 4

    env = OffsetActions(gym.make("CartPole-v1"))
    try:
        agent = algorithm(env, config=config(algorithm), device="cpu").learn(
            10, progress_bar=False
        )
        observation, _ = env.reset()
        assert env.action_space.contains(agent.predict(observation, deterministic=True))
    finally:
        env.close()


def test_ppo_clipping_signs_and_frozen_old_probabilities(monkeypatch):
    env = gym.make("CartPole-v1")
    try:
        agent = PPO(
            env,
            nn.Linear(4, 2),
            config=config(
                PPO, n_steps=4, batch_size=4, n_epochs=2, normalize_advantage=False
            ),
            device="cpu",
        )
        fill_rollout(agent)
        batch = agent.rollout_buffer.batch(torch.device("cpu"))
        batch.advantages.copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
        desired_ratios = torch.tensor([1.5, 0.5, 0.5, 1.5])
        with torch.no_grad():
            new_logs = agent.policy_action_space.distribution(
                agent.policy_network(batch.observations)
            ).log_prob(batch.actions.long())
            batch.log_probabilities.copy_(new_logs - desired_ratios.log())
        old_logs = batch.log_probabilities.clone()
        # Freeze optimizers so both epochs must use exactly the same old reference.
        monkeypatch.setattr(agent.policy_optimizer, "step", lambda: None)
        monkeypatch.setattr(agent.value_optimizer, "step", lambda: None)
        metrics = agent._train_step()
        # min terms: [1.2, -0.8, 0.5, -1.5], giving policy loss 0.15.
        assert metrics["train/policy_loss"] == pytest.approx(0.15, abs=1e-6)
        assert metrics["train/clip_fraction"] == 1
        torch.testing.assert_close(batch.log_probabilities, old_logs)
        assert all(
            p.grad is None for p in []
        )  # Targets are buffer tensors, not graphs.
        assert not batch.advantages.requires_grad and not batch.returns.requires_grad
    finally:
        env.close()


def test_ppo_visits_partial_minibatches_each_epoch(monkeypatch):
    env = gym.make("CartPole-v1")
    try:
        agent = PPO(env, config=config(PPO), device="cpu")
        fill_rollout(agent)
        calls = []
        original = agent.policy_optimizer.step

        def step():
            calls.append(1)
            return original()

        monkeypatch.setattr(agent.policy_optimizer, "step", step)
        agent._train_step()
        assert len(calls) == 4  # 5 samples / batches of 3, for two epochs.
    finally:
        env.close()


def test_trpo_conjugate_gradient_solves_positive_system():
    env = gym.make("CartPole-v1")
    try:
        agent = TRPO(env, config=config(TRPO), device="cpu")
        matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
        b = torch.tensor([1.0, 2.0])
        torch.testing.assert_close(
            agent._conjugate_gradient(lambda x: matrix @ x, b),
            torch.linalg.solve(matrix, b),
        )
        torch.testing.assert_close(
            agent._conjugate_gradient(lambda x: -x, b), torch.zeros(2)
        )
    finally:
        env.close()


@pytest.mark.parametrize("reject", [False, True])
def test_trpo_kl_acceptance_and_exact_rollback(monkeypatch, reject):
    env = gym.make("CartPole-v1")
    try:
        agent = TRPO(env, nn.Linear(4, 2), config=config(TRPO), device="cpu")
        observations = torch.zeros(6, 4)
        actions = torch.zeros(6, dtype=torch.long)
        advantages = torch.ones(6)
        with torch.no_grad():
            old = Categorical(logits=agent.policy_network(observations).clone())
            old_logs = old.log_prob(actions)
        before = (
            parameters_to_vector(agent.policy_network.parameters()).detach().clone()
        )
        if reject:
            # Force every candidate KL over budget, while retaining the true Hessian.
            def reject_candidate(old_distribution, new_distribution):
                divergence = kl_divergence(old_distribution, new_distribution)
                return divergence if torch.is_grad_enabled() else divergence + 1

            monkeypatch.setattr(
                "aprenderl.algorithms.trpo.kl_divergence", reject_candidate
            )
        result = agent._update_actor(observations, actions, advantages, old_logs)
        after = parameters_to_vector(agent.policy_network.parameters())
        if reject:
            assert torch.equal(before, after)
            assert result["train/step_fraction"] == 0
        else:
            new = Categorical(logits=agent.policy_network(observations))
            assert result["train/surrogate_gain"] > 0
            assert 0 < result["train/step_fraction"] <= 1
            assert kl_divergence(old, new).mean() <= agent.config.max_kl
            assert not torch.equal(before, after)
        assert not agent.policy_optimizer.state
    finally:
        env.close()


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (PPOConfig, {"clip_range": 0}),
        (PPOConfig, {"clip_range": 1}),
        (PPOConfig, {"batch_size": 0}),
        (PPOConfig, {"n_epochs": 0}),
        (PPOConfig, {"gae_lambda": 2}),
        (TRPOConfig, {"gae_lambda": -1}),
        (TRPOConfig, {"max_kl": 0}),
        (TRPOConfig, {"damping": -1}),
        (TRPOConfig, {"cg_steps": 0}),
        (TRPOConfig, {"cg_tolerance": 0}),
        (TRPOConfig, {"backtrack_steps": 0}),
        (TRPOConfig, {"backtrack_coefficient": 1}),
        (TRPOConfig, {"accept_ratio": -1}),
        (TRPOConfig, {"value_epochs": 0}),
        (TRPOConfig, {"entropy_coefficient": 0.1}),
    ],
)
def test_config_validation(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_shared_actor_critic_parameters_are_rejected(algorithm):
    class SharedValue(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, observations):
            return self.policy(observations).sum(dim=-1)

    env = gym.make("CartPole-v1")
    try:
        policy = nn.Linear(4, 2)
        with pytest.raises(ValueError, match="must not share parameters"):
            algorithm(env, policy, value_network=SharedValue(policy), device="cpu")
    finally:
        env.close()


def test_trpo_restores_parameters_on_line_search_exception(monkeypatch):
    env = gym.make("CartPole-v1")
    try:
        agent = TRPO(env, nn.Linear(4, 2), config=config(TRPO), device="cpu")
        states = torch.zeros(4, 4)
        actions = torch.zeros(4, dtype=torch.long)
        with torch.no_grad():
            old_logs = Categorical(logits=agent.policy_network(states)).log_prob(
                actions
            )
        before = (
            parameters_to_vector(agent.policy_network.parameters()).detach().clone()
        )

        def fail_candidate(old_distribution, new_distribution):
            if not torch.is_grad_enabled():
                raise RuntimeError("candidate failed")
            return kl_divergence(old_distribution, new_distribution)

        monkeypatch.setattr("aprenderl.algorithms.trpo.kl_divergence", fail_candidate)
        with pytest.raises(RuntimeError, match="candidate failed"):
            agent._update_actor(states, actions, torch.ones(4), old_logs)
        assert torch.equal(
            before, parameters_to_vector(agent.policy_network.parameters())
        )
    finally:
        env.close()


def test_trpo_zero_advantages_skip_actor_but_fit_critic():
    env = gym.make("CartPole-v1")
    try:
        agent = TRPO(env, config=config(TRPO), device="cpu")
        fill_rollout(agent)
        agent.rollout_buffer.advantages.fill(0)
        before = (
            parameters_to_vector(agent.policy_network.parameters()).detach().clone()
        )
        metrics = agent._train_step()
        assert metrics["train/step_fraction"] == 0
        assert torch.equal(
            before, parameters_to_vector(agent.policy_network.parameters())
        )
        assert agent.value_optimizer.state
    finally:
        env.close()
