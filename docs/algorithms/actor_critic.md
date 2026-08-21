# Actor-Critic

[Back to the algorithm index](../algorithms.md)

**Policy classification:** On-policy. Each update uses a small rollout sampled
from the current stochastic policy and does not replay old experience.

**Use when:** observations are numeric `Box` vectors or tensors, actions are
either discrete or a finite or fully unbounded continuous `Box`, and an
incremental policy-gradient method is preferable to waiting for complete
episodes as REINFORCE does.

## Actor and critic

Actor-Critic learns two functions with separate neural networks:

- the actor $\pi_\theta(a\mid s)$, a categorical policy for discrete actions or
  a plain/squashed diagonal-Gaussian policy for continuous actions;
- the critic $V_\phi(s)$, an estimate of the expected return from a state.

The default networks flatten each observation and pass it through two 128-unit
ReLU layers. A discrete actor outputs one logit per action. A continuous actor
outputs Gaussian means and holds one learned log standard deviation per action
dimension. `tanh` and affine scaling keep finite actions inside their Gymnasium
bounds, while fully unbounded spaces use the raw Gaussian. The critic outputs
one scalar per observation.

## One-step TD target

For every transition $(s_t,a_t,r_{t+1},s_{t+1})$, the critic constructs the
target

$$
y_t = r_{t+1} + \gamma(1-d_t)V_\phi(s_{t+1}),
$$

where $d_t=1$ only when Gymnasium reports `terminated`. A time-limit
`truncated` transition still bootstraps because the underlying Markov decision
process may continue beyond that external limit.

The critic minimizes squared TD error:

$$
\mathcal{L}_{\mathrm{value}}(\phi)
= \left(V_\phi(s_t)-y_t\right)^2.
$$

The target is detached during optimization, so the critic changes only through
its current-state estimate.

## Actor update

The same one-step TD error provides the advantage estimate

$$
\hat{A}_t = y_t - V_\phi(s_t).
$$

The actor minimizes

$$
\mathcal{L}_{\mathrm{policy}}(\theta)
= -\log\pi_\theta(a_t\mid s_t)\,\operatorname{stopgrad}(\hat{A}_t).
$$

A positive advantage increases the probability of the sampled action, while a
negative advantage decreases it. Detaching the advantage keeps the actor loss
from changing the critic. An optional entropy bonus encourages broader action
distributions:

$$
\mathcal{L}_{\mathrm{actor}}(\theta)
= \mathcal{L}_{\mathrm{policy}}(\theta)
- \beta\mathcal{H}\left(\pi_\theta(\cdot\mid s_t)\right).
$$

The algorithm collects `n_steps` transitions before averaging these actor and
critic losses into one update. This small on-policy batch reduces variance, but
every target remains the one-step target above; trajectories are not replayed.
Adam updates the two networks independently using `learning_rate` and
`value_learning_rate`, and both gradient norms are clipped to `max_grad_norm`.
This single-environment implementation is not the parallel-environment A2C
variant. AprendeRL's separate [A2C implementation](a2c.md) adds multi-step GAE
targets while retaining the library's intentionally small, single-environment
training loop.

## Usage

```python
import gymnasium as gym

from aprenderl import ActorCritic, ActorCriticConfig

env = gym.make("CartPole-v1")
config = ActorCriticConfig(
    learning_rate=1e-3,
    value_learning_rate=1e-3,
    n_steps=32,
    entropy_coefficient=1e-3,
)
agent = ActorCritic(env, config=config).learn(20_000)
```

No algorithm switch is needed for continuous actions:

```python
env = gym.make("Pendulum-v1")
config = ActorCriticConfig(
    learning_rate=3e-4,
    value_learning_rate=1e-3,
    n_steps=64,
)
agent = ActorCritic(env, config=config).learn(50_000)
```

Custom networks can be supplied as `policy_network=...` and
`value_network=...`. A discrete policy returns shape
`(batch_size, action_count)`. A continuous policy returns a
`(means, log_stds)` tuple with both tensors shaped
`(batch_size, flattened_action_size)`. The value network returns
`(batch_size,)`. Continuous entropy uses the diagonal-Gaussian entropy; for
finite actions this is the pre-squash entropy.

The implementation is in
[`src/aprenderl/algorithms/actor_critic.py`](../../src/aprenderl/algorithms/actor_critic.py),
and the runnable example is
[`examples/train_actor_critic.ipynb`](../../examples/train_actor_critic.ipynb).
The continuous example is
[`examples/train_actor_critic_continuous.ipynb`](../../examples/train_actor_critic_continuous.ipynb).
