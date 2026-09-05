# PPO

[Back to the algorithm index](../algorithms.md)

**Policy classification:** On-policy. PPO-Clip reuses each fresh rollout for
several shuffled minibatch passes, then discards it.

**Use when:** you want GAE-based policy learning with ordinary first-order
optimizers and a clipped surrogate to moderate policy changes.

## Clipped surrogate

Freeze the collecting policy $\theta_{old}$. For its sampled state-action
pairs, define the probability ratio

$$\rho_t(\theta)=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_{old}}(a_t\mid s_t)).$$

PPO minimizes

$$L_\pi=-\frac1N\sum_t\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)
-c_H\frac1N\sum_t H(\pi_\theta(\cdot\mid s_t)).$$

Here $\hat A_t$ is the fixed advantage, $\epsilon=$ `clip_range`, and
$c_H=$ `entropy_coefficient`. $H$ is categorical or Gaussian entropy. For
bounded actions the library uses pre-squash Gaussian entropy as a proxy.
The negative sign converts surrogate maximization into loss minimization.

For positive advantages, gains from increasing the ratio stop above
$1+\epsilon$. For negative advantages, gains from decreasing it stop below
$1-\epsilon$. Harmful changes still receive a penalty. Clipping is not a hard
constraint on ratios or KL, and does not guarantee return improvement.

`n_epochs` controls passes through each rollout. Every epoch reshuffles the
indices and consumes all samples, including a final partial minibatch.
`batch_size` may exceed rollout size, producing one full-batch step per epoch.
Old log probabilities, GAE advantages, and value targets stay fixed throughout.
Actor and critic use separate Adam optimizers and gradient-norm clipping via
`max_grad_norm`. There is no value clipping or KL early stopping.

## Settings and diagnostics

Defaults are `learning_rate=3e-4`, `value_learning_rate=1e-3`, `n_steps=1024`,
`batch_size=64`, `n_epochs=10`, `clip_range=0.2`, `gamma=0.99`,
`gae_lambda=0.95`, `normalize_advantage=True`, `entropy_coefficient=0`, and
`max_grad_norm=0.5`.

Reported losses and entropy are measured on the full rollout after all epochs.
`train/clip_fraction` is the fraction of sampled ratios outside the clipping
interval, not the fraction with zero gradient. `train/approx_kl` averages
$(\rho_t-1)-\log\rho_t$, a sampled estimate of old-to-new KL. Larger learning
rates or more epochs can increase policy change even with clipping.

## Rollouts and value targets

Both the actor $\pi_\theta(a\mid s)$ and critic $V_\phi(s)$ are separate neural
networks. At transition $t$, $s_t$ is the observation, $a_t$ the sampled action,
$r_{t+1}$ the reward, $\gamma$ the discount, and $\lambda$ the GAE trace decay:

$$
\delta_t=r_{t+1}+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t),\qquad
\hat A_t=\delta_t+\gamma\lambda(1-b_t)\hat A_{t+1}.
$$

Here $d_t$ is one only for `terminated`, while $b_t$ is one for either
`terminated` or `truncated`. Truncation retains the bootstrap value of the
final observation but stops the trace before the environment resets. A rollout
cutoff also bootstraps, with the next advantage initialized to zero.

Critic targets are $y_t=\hat A_t+V_\phi(s_t)$. They remain fixed during fitting:

$$L_V(\phi)=\frac1N\sum_t(V_\phi(s_t)-y_t)^2,$$

where $N$ is rollout size. `normalize_advantage=True` standardizes actor
advantages across the rollout when it has more than one sample. It does not
normalize critic targets. Gradients never flow through old policy quantities,
advantages, or targets.

## API and environment support

The public methods are `learn(total_timesteps, progress_bar=True)`,
`predict(observation, deterministic=False)`, `save(path)`, and
`load(path, env, ...)`. Constructor settings go in the configuration class.
Callbacks and logging use the common `BaseAlgorithm` lifecycle.

Use numeric `Box` observations and either `Discrete` actions (including
nonzero `start`) or floating-point `Box` actions. Finite continuous bounds use
a tanh-squashed diagonal Gaussian with affine rescaling; fully unbounded
bounds use a plain diagonal Gaussian. Mixed or one-sided bounds are rejected.
Exploration comes from policy sampling. Deterministic prediction selects the
categorical mode or transformed Gaussian mean.

Custom `policy_network` modules return categorical logits of shape
`(batch, action_count)`, or `(means, log_stds)` tensors of shape
`(batch, flattened_action_size)` for continuous actions. `value_network`
returns `(batch,)`. Actor and critic may not share parameters. Networks run
in evaluation mode, with gradients enabled for updates, so dropout and batch
normalization do not change the collecting-policy reference. Custom forward
methods must also avoid changing state or adding randomness of their own.

Collection uses one environment and updates after each complete `n_steps`
rollout. `num_updates` counts completed rollout updates, not optimizer steps.
An unfinished rollout remains in memory for the next `learn` call. Checkpoints
save network weights, optimizer states, configuration, action-space metadata,
and training counters; they do not save pending rollouts, environment state,
or random-number-generator state. Reloading starts fresh collection rather
than reproducing the exact trajectory. Custom-network checkpoints require
matching modules on load. Observation shapes and action shapes/bounds/offsets
are checked against the supplied environment.

## Example

```python
import gymnasium as gym
from aprenderl import PPO, PPOConfig

env = gym.make("CartPole-v1")
try:
    agent = PPO(env, config=PPOConfig(n_steps=256, seed=7), device="cpu")
    agent.learn(20_000)
    agent.save("artifacts/ppo.pt")
finally:
    env.close()
```

## Continuous-action example

```python
import gymnasium as gym
from aprenderl import PPO, PPOConfig

env = gym.make("Pendulum-v1")
try:
    agent = PPO(env, config=PPOConfig(n_steps=1024, seed=7), device="cpu")
    agent.learn(51_200)
    observation, _ = env.reset(seed=42)
    action = agent.predict(observation, deterministic=True)
    assert env.action_space.contains(action)
    agent.save("artifacts/ppo_continuous.pt")
    restored = PPO.load("artifacts/ppo_continuous.pt", env, device="cpu")
finally:
    env.close()
```

The default `GaussianPolicyNetwork` learns both means and log standard
deviations. Actions retain the environment's shape and floating dtype, including
multidimensional boxes; log densities sum over all flattened action dimensions.
Deterministic actions use the transformed mean, not a random sample.

The [continuous-action notebook](../../examples/ppo_continuous.ipynb)
uses lower-gravity Pendulum for a compact CPU demonstration, plots episode
returns, and renders deterministic evaluation in a separate environment.

See the [source](../../src/aprenderl/algorithms/ppo.py),
[public-API example](../../examples/ppo.ipynb), and
[from-scratch study notebook](../../study/06_policy_gradients/07_ppo.ipynb).

## Reference

John Schulman et al. (2017),
[*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347).
