# A2C

[Back to the algorithm index](../algorithms.md)

**Policy classification:** On-policy. Each update uses one fresh, fixed-length
rollout sampled from the current stochastic policy.

**Use when:** observations are numeric `Box` vectors or tensors, actions are
either discrete or a finite or fully unbounded continuous `Box`, and reward
information should propagate across multiple rollout steps without waiting for
complete episodes.

## Relationship to Actor-Critic

A2C learns the same two functions as AprendeRL's one-step Actor-Critic:

- an actor $\pi_\theta(a\mid s)$ that defines a categorical distribution for
  discrete actions or a plain/squashed diagonal Gaussian for continuous
  actions;
- a critic $V_\phi(s)$ that estimates the expected return from a state.

The difference is the learning signal. Actor-Critic uses a separate one-step TD
error at every stored transition. A2C combines consecutive TD errors with
generalized advantage estimation (GAE), allowing later rewards in the rollout
to influence earlier actions.

This educational implementation uses one environment worker. That is the
single-worker form of the synchronous A2C update: collect `n_steps`, calculate
one batch of targets, update once, then discard the rollout. It deliberately
does not add vector-environment infrastructure.

## Generalized advantage estimation

For transition $t$, define the TD residual

$$
\delta_t = r_{t+1} + \gamma(1-d_t)V_\phi(s_{t+1}) - V_\phi(s_t),
$$

where $d_t=1$ only for a true Gymnasium `terminated` state. The advantage is
computed backward through the rollout:

$$
\hat A_t = \delta_t
+ \gamma\lambda(1-b_t)\hat A_{t+1},
$$

where $b_t=1$ at either `terminated` or `truncated` episode boundaries. Thus a
time-limit truncation still bootstraps through $V(s_{t+1})$, but the recursion
does not leak information from the reset episode into the preceding episode.

`gae_lambda` controls the bias-variance tradeoff. A value of zero recovers the
one-step TD advantage; a value of one uses the longest available multi-step
estimate within the rollout. The default is `1.0`.

The critic regression target is

$$
\hat R_t = \hat A_t + V_\phi(s_t).
$$

## Losses

For a rollout containing $N$ transitions, A2C minimizes the policy loss

$$
\mathcal L_{\mathrm{policy}}
= -\frac{1}{N}\sum_t
\log\pi_\theta(a_t\mid s_t)\operatorname{stopgrad}(\hat A_t),
$$

and value loss

$$
\mathcal L_{\mathrm{value}}
= \frac{1}{N}\sum_t\left(V_\phi(s_t)-\hat R_t\right)^2.
$$

Set `normalize_advantage=True` to standardize the policy weights within each
rollout. The value targets are never normalized. The reported combined loss is

$$
\mathcal L = \mathcal L_{\mathrm{policy}}
- \beta\mathcal H(\pi_\theta)
+ c_v\mathcal L_{\mathrm{value}},
$$

where `entropy_coefficient` is $\beta$ and `value_loss_coefficient` is $c_v$.
Because the actor and critic are separate networks, AprendeRL applies this as
two explicit Adam updates and clips both gradient norms to `max_grad_norm`.

## Usage

```python
import gymnasium as gym

from aprenderl import A2C, A2CConfig

env = gym.make("CartPole-v1")
config = A2CConfig(
    learning_rate=7e-4,
    value_learning_rate=7e-4,
    gamma=0.99,
    n_steps=32,
    gae_lambda=0.95,
    normalize_advantage=False,
)
agent = A2C(env, config=config).learn(20_000)
```

The same implementation handles continuous actions:

```python
env = gym.make("Pendulum-v1")
config = A2CConfig(
    learning_rate=3e-4,
    value_learning_rate=1e-3,
    n_steps=64,
    gae_lambda=0.95,
    normalize_advantage=True,
)
agent = A2C(env, config=config).learn(50_000)
```

Custom networks can be supplied as `policy_network=...` and
`value_network=...`. A discrete policy returns shape
`(batch_size, action_count)`. A continuous policy returns a
`(means, log_stds)` tuple with both tensors shaped
`(batch_size, flattened_action_size)`. The value network returns
`(batch_size,)`. Continuous entropy uses the diagonal-Gaussian entropy; for
finite actions this is the pre-squash entropy. Checkpoints include both
networks, both optimizers, the action bounds, the A2C configuration, and
training counters.

The implementation is in
[`src/aprenderl/algorithms/a2c.py`](../../src/aprenderl/algorithms/a2c.py), and
the runnable example is
[`examples/train_a2c.ipynb`](../../examples/train_a2c.ipynb). The continuous
example is
[`examples/train_a2c_continuous.ipynb`](../../examples/train_a2c_continuous.ipynb).

## Foundational papers

- Volodymyr Mnih et al. (2016),
  [*Asynchronous Methods for Deep Reinforcement Learning*](https://proceedings.mlr.press/v48/mniha16.html),
  ICML 2016. Describes A3C, from which the synchronous A2C variant is derived.
- John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter
  Abbeel (2016),
  [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438),
  ICLR 2016. Introduces GAE.
