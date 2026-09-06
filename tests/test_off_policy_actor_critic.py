"""Analytic updates, action semantics, lifecycle and checkpoint regression tests."""

import copy
import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from aprenderl import (
    DDPG,
    SAC,
    TD3,
    DDPGConfig,
    DiscreteSAC,
    DiscreteSACConfig,
    OffPolicyAlgorithm,
    OnPolicyAlgorithm,
    PolicyGradientAlgorithm,
    SACConfig,
    TD3Config,
)
from aprenderl.buffers import ReplayBatch, ReplayBuffer
from aprenderl.callbacks import BaseCallback
from aprenderl.distributions import SquashedGaussianDistribution
from aprenderl.networks import (
    ContinuousQNetwork,
    DeterministicPolicyNetwork,
    SACPolicyNetwork,
)

ALGORITHMS = [
    (DDPG, DDPGConfig),
    (TD3, TD3Config),
    (SAC, SACConfig),
    (DiscreteSAC, DiscreteSACConfig),
]


class ControlEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def __init__(self, discrete=False, truncated=False):
        self.action_space = (
            gym.spaces.Discrete(2, start=3)
            if discrete
            else gym.spaces.Box(
                np.array([[-2.0, 1.0]], dtype=np.float32),
                np.array([[4.0, 5.0]], dtype=np.float32),
            )
        )
        self.truncated = truncated

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)
        return np.ones(2, dtype=np.float32), 1.0, not self.truncated, self.truncated, {}


def make_agent(cls, config_cls, **kwargs):
    config = config_cls(
        batch_size=2, buffer_size=20, learning_starts=0, seed=7, **kwargs
    )
    return cls(ControlEnv(cls is DiscreteSAC), config=config, device="cpu")


def batch(continuous=True):
    return ReplayBatch(
        torch.zeros(3, 2),
        torch.zeros(
            3,
            2 if continuous else 1,
            dtype=torch.float32 if continuous else torch.int64,
        ),
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.ones(3, 2),
        torch.tensor([[0.0], [1.0], [0.0]]),
        torch.tensor([[0.0], [0.0], [1.0]]),
    )


class ConstantCritic(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.last_actions = None

    def forward(self, observations, actions=None):
        self.last_actions = actions
        return self.values.expand(len(observations), -1)


class ConstantPolicy(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, observations):
        return self.values.expand(len(observations), -1)


@pytest.mark.parametrize("cls,config_cls", ALGORITHMS)
def test_api_action_shape_offset_and_replay(cls, config_cls):
    agent = make_agent(cls, config_cls)
    assert isinstance(agent, OffPolicyAlgorithm)
    assert isinstance(agent, PolicyGradientAlgorithm)
    assert not isinstance(agent, OnPolicyAlgorithm)
    observation, _ = agent.env.reset()
    for deterministic in (False, True):
        assert agent.env.action_space.contains(
            agent.predict(observation, deterministic=deterministic)
        )
    agent.learn(4, progress_bar=False)
    assert agent.num_timesteps == 4 and agent.num_updates == 3
    assert len(agent.episode_returns) == 4
    if cls is DiscreteSAC:
        assert set(agent.replay_buffer.actions[:4, 0]) <= {0, 1}
    else:
        assert agent.replay_buffer.actions.dtype == np.float32
        assert agent.replay_buffer.actions.shape == (20, 2)
    with pytest.raises(ValueError, match="observation shape"):
        agent.predict(np.zeros(3))


def test_ddpg_target_uses_target_actor_and_bootstraps_truncation():
    agent = make_agent(DDPG, DDPGConfig, gamma=0.5)
    agent.target_policy_network = ConstantPolicy([0.0, 0.0])
    agent.target_critics = nn.ModuleList([ConstantCritic([4.0])])
    targets = agent._td_target(batch())
    torch.testing.assert_close(targets, torch.tensor([[3.0], [2.0], [5.0]]))
    assert not targets.requires_grad
    torch.testing.assert_close(
        agent.target_critics[0].last_actions, torch.tensor([[1.0, 3.0]]).expand(3, -1)
    )


def test_td3_target_minimum_and_normalized_noise_clipping(monkeypatch):
    agent = make_agent(TD3, TD3Config, gamma=0.5)
    agent.target_policy_network = ConstantPolicy([0.9, -0.9])
    agent.target_critics = nn.ModuleList([ConstantCritic([8.0]), ConstantCritic([4.0])])
    monkeypatch.setattr(
        torch, "randn_like", lambda x: x.new_tensor([10.0, -10.0]).expand_as(x)
    )
    torch.testing.assert_close(
        agent._td_target(batch()), torch.tensor([[3.0], [2.0], [5.0]])
    )
    torch.testing.assert_close(
        agent.target_critics[0].last_actions, torch.tensor([[4.0, 1.0]]).expand(3, -1)
    )
    agent.target_policy_network = ConstantPolicy([0.0, 0.0])
    agent._td_target(batch())
    torch.testing.assert_close(
        agent.target_critics[0].last_actions, torch.tensor([[2.5, 2.0]]).expand(3, -1)
    )


def test_sac_target_includes_entropy_and_terminal_mask(monkeypatch):
    agent = make_agent(SAC, SACConfig, gamma=0.5, ent_coef=0.2)
    agent.target_critics = nn.ModuleList([ConstantCritic([8.0]), ConstantCritic([4.0])])
    monkeypatch.setattr(
        agent,
        "_sample_policy",
        lambda obs: (torch.zeros(len(obs), 2), torch.full((len(obs), 1), -2.0)),
    )
    torch.testing.assert_close(
        agent._td_target(batch()), torch.tensor([[3.2], [2.0], [5.2]])
    )


def test_discrete_sac_exact_expectation():
    agent = make_agent(DiscreteSAC, DiscreteSACConfig, gamma=0.5, ent_coef=0.2)
    agent.policy_network = ConstantPolicy([math.log(0.25), math.log(0.75)])
    agent.target_critics = nn.ModuleList(
        [ConstantCritic([8.0, 2.0]), ConstantCritic([4.0, 6.0])]
    )
    entropy = -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
    value = 0.25 * 4 + 0.75 * 2 + 0.2 * entropy
    expected = torch.tensor([[1 + 0.5 * value], [2.0], [3 + 0.5 * value]])
    torch.testing.assert_close(agent._td_target(batch(False)), expected)


@pytest.mark.parametrize("cls,config_cls", ALGORITHMS)
def test_gradient_flow_and_polyak_update(cls, config_cls):
    agent = make_agent(cls, config_cls, tau=0.25)
    before_policy = copy.deepcopy(agent.policy_network.state_dict())
    before_target = copy.deepcopy(agent.target_critics.state_dict())
    agent.learn(2, progress_bar=False)
    updated = cls is not TD3
    assert (
        any(
            not torch.equal(before_policy[k], v)
            for k, v in agent.policy_network.state_dict().items()
        )
        == updated
    )
    for key, value in agent.target_critics.state_dict().items():
        expected = (
            0.75 * before_target[key] + 0.25 * agent.critic_networks.state_dict()[key]
            if updated
            else before_target[key]
        )
        torch.testing.assert_close(value, expected)
    assert all(p.grad is None for p in agent.target_critics.parameters())
    assert all(p.grad is None for p in agent.critic_networks.parameters())
    if cls is TD3:
        agent.learn(1, progress_bar=False)
        assert any(
            not torch.equal(before_policy[k], v)
            for k, v in agent.policy_network.state_dict().items()
        )


def test_ddpg_actor_chain_rule():
    class LinearCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.slope = nn.Parameter(torch.tensor([2.0, -1.0]))

        def forward(self, observations, actions):
            return (actions * self.slope).sum(-1, keepdim=True)

    actor = ConstantPolicy([0.0, 0.0])
    critic = LinearCritic()
    agent = DDPG(
        ControlEnv(),
        actor,
        critic_networks=[critic],
        config=DDPGConfig(
            batch_size=2,
            buffer_size=2,
            learning_starts=0,
            critic_learning_rate=1e-30,
            max_grad_norm=100,
        ),
        device="cpu",
    )
    agent.learn(2, progress_bar=False)
    # Environment half-ranges [3, 2] multiply dQ/da [2, -1].
    torch.testing.assert_close(actor.values.grad, torch.tensor([-6.0, 2.0]))


@pytest.mark.parametrize(
    "cls,config_cls", [(SAC, SACConfig), (DiscreteSAC, DiscreteSACConfig)]
)
@pytest.mark.parametrize("target,sign", [(-100.0, -1), (100.0, 1)])
def test_temperature_moves_toward_entropy_target(cls, config_cls, target, sign):
    agent = make_agent(cls, config_cls)
    agent.target_entropy = (
        target  # Isolate update direction, even beyond feasible entropy.
    )
    before = agent.ent_coef.item()
    agent.learn(2, progress_bar=False)
    assert (agent.ent_coef.item() - before) * sign > 0


@pytest.mark.parametrize(
    "cls,config_cls", [(SAC, SACConfig), (DiscreteSAC, DiscreteSACConfig)]
)
def test_fixed_temperature(cls, config_cls):
    agent = make_agent(cls, config_cls, ent_coef=0.0)
    agent.learn(3, progress_bar=False)
    assert agent.ent_coef.item() == 0 and agent.ent_coef_optimizer is None


@pytest.mark.parametrize("cls,config_cls", ALGORITHMS)
def test_checkpoint_roundtrip_and_resume(cls, config_cls, tmp_path):
    agent = make_agent(cls, config_cls)
    agent.learn(5, progress_bar=False)
    observation, _ = agent.env.reset()
    action = agent.predict(observation, deterministic=True)
    path = tmp_path / "agent.pt"
    agent.save(path)
    restored = cls.load(path, ControlEnv(cls is DiscreteSAC), device="cpu")
    np.testing.assert_array_equal(
        action, restored.predict(observation, deterministic=True)
    )
    assert restored.config == agent.config
    assert restored.num_updates == agent.num_updates
    assert restored.episode_returns == agent.episode_returns
    assert len(restored.replay_buffer) == 0
    for k, v in agent.target_critics.state_dict().items():
        torch.testing.assert_close(v, restored.target_critics.state_dict()[k])
    if cls in (SAC, DiscreteSAC):
        torch.testing.assert_close(agent.ent_coef, restored.ent_coef)
        assert restored.ent_coef_optimizer.state_dict()["state"]
    restored.learn(3, progress_bar=False)
    assert restored.num_updates == agent.num_updates + 2
    incompatible = ControlEnv(cls is DiscreteSAC)
    if cls is DiscreteSAC:
        incompatible.action_space = gym.spaces.Discrete(2, start=0)
    else:
        incompatible.action_space = gym.spaces.Box(-1.0, 1.0, (1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="spaces"):
        cls.load(path, incompatible, device="cpu")
    incompatible.observation_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
    with pytest.raises(ValueError, match="spaces"):
        cls.load(path, incompatible, device="cpu")


@pytest.mark.parametrize("cls,config_cls", ALGORITHMS)
def test_custom_network_checkpoints(cls, config_cls, tmp_path):
    env = ControlEnv(cls is DiscreteSAC)
    if cls is DiscreteSAC:
        actor = nn.Linear(2, 2)
        critics = [nn.Linear(2, 2), nn.Linear(2, 2)]
    else:
        factory = SACPolicyNetwork if cls is SAC else DeterministicPolicyNetwork
        actor = factory(2, 2, (8,))
        critics = [ContinuousQNetwork(2, 2, (8,)) for _ in range(cls.n_critics)]
    agent = cls(
        env,
        actor,
        critic_networks=critics,
        config=config_cls(batch_size=2, buffer_size=10, learning_starts=0),
        device="cpu",
    )
    agent.learn(3, progress_bar=False)
    path = tmp_path / "custom.pt"
    agent.save(path)
    with pytest.raises(ValueError, match="policy_network"):
        cls.load(path, env, device="cpu")
    with pytest.raises(ValueError, match="critic_networks"):
        cls.load(path, env, policy_network=copy.deepcopy(actor), device="cpu")
    restored = cls.load(
        path,
        env,
        policy_network=copy.deepcopy(actor),
        critic_networks=copy.deepcopy(critics),
        device="cpu",
    )
    restored.learn(3, progress_bar=False)


@pytest.mark.parametrize("cls,config_cls", ALGORITHMS)
def test_environment_validation(cls, config_cls):
    env = ControlEnv(cls is DiscreteSAC)
    env.observation_space = gym.spaces.Discrete(3)
    with pytest.raises(TypeError, match="observation"):
        cls(env, device="cpu")
    env = ControlEnv(cls is not DiscreteSAC)
    with pytest.raises(TypeError, match="action space"):
        cls(env, device="cpu")
    if cls is not DiscreteSAC:
        for space in [
            gym.spaces.Box(-np.inf, np.inf, (2,)),
            gym.spaces.Box(0, 0, (2,)),
            gym.spaces.Box(0, 2, (2,), dtype=np.int32),
            gym.spaces.Box(-1, 1, ()),
        ]:
            env.action_space = space
            with pytest.raises(ValueError):
                cls(env, device="cpu")


@pytest.mark.parametrize("cls,config_cls", ALGORITHMS)
def test_reject_invalid_custom_networks(cls, config_cls):
    env = ControlEnv(cls is DiscreteSAC)
    with pytest.raises(ValueError, match="policy_network"):
        cls(env, nn.Linear(2, 3), device="cpu")
    with pytest.raises(ValueError, match="critic"):
        cls(env, critic_networks=[nn.Linear(2, 3)] * cls.n_critics, device="cpu")
    with pytest.raises(ValueError, match="contain"):
        cls(env, critic_networks=[], device="cpu")


@pytest.mark.parametrize(
    "config_cls", [DDPGConfig, TD3Config, SACConfig, DiscreteSACConfig]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": float("nan")},
        {"tau": 0},
        {"batch_size": 0},
        {"learning_rate": float("inf")},
        {"learning_starts": -1},
        {"gradient_steps": 1.5},
        {"buffer_size": 1},
        {"train_freq": 0},
    ],
)
def test_config_validation(config_cls, kwargs):
    with pytest.raises(ValueError):
        config_cls(**kwargs)


def test_algorithm_specific_config_validation():
    for cls, kwargs in [
        (DDPGConfig, {"action_noise": -1}),
        (TD3Config, {"policy_delay": 0}),
        (TD3Config, {"target_policy_noise": float("nan")}),
        (TD3Config, {"target_noise_clip": -1}),
        (SACConfig, {"ent_coef": "bad"}),
        (SACConfig, {"ent_coef": -1}),
        (SACConfig, {"initial_ent_coef": 0}),
        (SACConfig, {"target_entropy": float("inf")}),
    ]:
        with pytest.raises(ValueError):
            cls(**kwargs)
    with pytest.raises(ValueError, match="target_entropy"):
        make_agent(DiscreteSAC, DiscreteSACConfig, target_entropy=2)


@pytest.mark.parametrize("cls,config_cls", ALGORITHMS)
def test_warmup_schedule_callbacks_and_truncation(cls, config_cls):
    class Stop(BaseCallback):
        def on_step(self, algorithm, transition):
            return algorithm.num_timesteps < 8

    agent = make_agent(cls, config_cls, train_freq=2, gradient_steps=3)
    agent.env.truncated = True
    agent.callback = Stop()
    agent.learn(20, progress_bar=False)
    assert agent.num_timesteps == 8 and agent.num_updates == 12
    assert agent.replay_buffer.terminated[:8].sum() == 0
    assert agent.replay_buffer.truncated[:8].sum() == 8
    warmup = cls(
        ControlEnv(cls is DiscreteSAC),
        config=config_cls(batch_size=2, learning_starts=5),
        device="cpu",
    )
    warmup.predict = lambda *args, **kwargs: pytest.fail("policy used during warmup")
    warmup.learn(4, progress_bar=False)
    assert warmup.num_updates == 0


def test_continuous_replay_preserves_fractional_multidimensional_actions():
    replay = ReplayBuffer(2, (2,), action_shape=(2, 2), seed=0)
    action = np.array([[0.25, -0.75], [1.5, 2.5]], dtype=np.float32)
    replay.add(np.zeros(2), action, 1, np.ones(2), False, True)
    sample = replay.sample(3, torch.device("cpu"))
    torch.testing.assert_close(sample.actions, torch.tensor(action).expand(3, 2, 2))
    assert sample.truncated.sum() == 3 and sample.terminated.sum() == 0


def test_squashed_rsample_density_gradient_and_saturation():
    means = torch.tensor([[0.2, -0.3]], requires_grad=True)
    log_stds = torch.tensor([[-0.5, -0.5]], requires_grad=True)
    low, high = torch.tensor([-2.0, 1.0]), torch.tensor([4.0, 5.0])
    dist = SquashedGaussianDistribution(means, log_stds, low, high)
    torch.manual_seed(7)
    actions, log_prob = dist.rsample_with_log_prob()
    torch.testing.assert_close(log_prob, dist.log_prob(actions), atol=2e-5, rtol=2e-5)
    (actions.sum() + log_prob.sum()).backward()
    assert means.grad.abs().sum() > 0 and log_stds.grad.abs().sum() > 0
    means = torch.tensor([[100.0, -100.0]], requires_grad=True)
    dist = SquashedGaussianDistribution(means, torch.zeros_like(means), low, high)
    actions, log_prob = dist.rsample_with_log_prob()
    assert torch.isfinite(log_prob).all()
    log_prob.sum().backward()
    assert torch.isfinite(means.grad).all()


def test_sac_entropy_target_accounts_for_action_scale():
    agent = make_agent(SAC, SACConfig)
    assert agent.target_entropy == pytest.approx(-2 + math.log(6))


def test_checkpoint_rejects_other_algorithm(tmp_path):
    agent = make_agent(DDPG, DDPGConfig)
    path = tmp_path / "ddpg.pt"
    agent.save(path)
    with pytest.raises(ValueError, match="algorithm"):
        TD3.load(path, agent.env, device="cpu")


def test_custom_batchnorm_validation_and_frozen_statistics():
    actor = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2), nn.Tanh())
    agent = DDPG(
        ControlEnv(),
        actor,
        config=DDPGConfig(batch_size=2, buffer_size=10, learning_starts=0),
        device="cpu",
    )
    before = actor[0].running_mean.clone()
    agent.learn(3, progress_bar=False)
    torch.testing.assert_close(actor[0].running_mean, before)
    assert not actor.training and not agent.target_policy_network.training


def test_discrete_actor_gradient_is_exact_policy_expectation():
    actor = ConstantPolicy([0.0, 0.0])
    critics = [ConstantCritic([1.0, 3.0]), ConstantCritic([2.0, 4.0])]
    agent = DiscreteSAC(
        ControlEnv(True),
        actor,
        critic_networks=critics,
        config=DiscreteSACConfig(
            batch_size=2,
            buffer_size=10,
            learning_starts=0,
            ent_coef=0.0,
            critic_learning_rate=1e-30,
        ),
        device="cpu",
    )
    agent.learn(2, progress_bar=False)
    # -sum softmax(logits) * [1, 3] has gradient [0.5, -0.5] at equal logits.
    torch.testing.assert_close(actor.values.grad, torch.tensor([0.5, -0.5]))


def test_sac_actor_gradient_flows_through_sampled_action(monkeypatch):
    class GaussianActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.means = nn.Parameter(torch.zeros(2))

        def forward(self, observations):
            means = self.means.expand(len(observations), -1)
            return means, torch.full_like(means, -20.0)

    class LinearCritic(nn.Module):
        def __init__(self, slope):
            super().__init__()
            self.slope = nn.Parameter(torch.tensor(slope))

        def forward(self, observations, actions):
            return self.slope * actions[:, :1]

    actor = GaussianActor()
    agent = SAC(
        ControlEnv(),
        actor,
        critic_networks=[LinearCritic(2.0), LinearCritic(3.0)],
        config=SACConfig(
            batch_size=2,
            buffer_size=10,
            learning_starts=0,
            ent_coef=0.0,
            critic_learning_rate=1e-30,
            max_grad_norm=100,
        ),
        device="cpu",
    )
    agent.learn(2, progress_bar=False)
    # First action is 1 + 3*tanh(mean); min critic has slope 2 at action 1.
    torch.testing.assert_close(actor.means.grad, torch.tensor([-6.0, 0.0]))


def test_reject_shared_critic_parameters():
    critic = ContinuousQNetwork(2, 2)
    with pytest.raises(ValueError, match="disjoint"):
        TD3(ControlEnv(), critic_networks=[critic, critic], device="cpu")


@pytest.mark.parametrize("cls,config_cls", [(DDPG, DDPGConfig), (TD3, TD3Config)])
def test_target_actor_polyak_and_behavior_rng_roundtrip(cls, config_cls, tmp_path):
    agent = make_agent(cls, config_cls, tau=0.1)
    before = copy.deepcopy(agent.target_policy_network.state_dict())
    agent.learn(2 if cls is DDPG else 3, progress_bar=False)
    for key, value in agent.target_policy_network.state_dict().items():
        torch.testing.assert_close(
            value, 0.9 * before[key] + 0.1 * agent.policy_network.state_dict()[key]
        )
    path = tmp_path / "noise.pt"
    agent.save(path)
    restored = cls.load(path, ControlEnv(), device="cpu")
    observation, _ = agent.env.reset()
    np.testing.assert_array_equal(
        agent.predict(observation), restored.predict(observation)
    )
