# TRPO

[Back to the algorithm index](../algorithms.md)

**Policy classification:** On-policy. Each fresh rollout supplies one
constrained actor update and repeated critic regression steps.

**Use when:** studying natural gradients, conjugate gradients, and a practical
trust-region policy update with a measured mean-KL budget.

## Surrogate and trust region

Freeze the collecting policy $\theta_{old}$ and estimated advantages
$\hat A_t$. With $N$ rollout transitions, maximize

$$L(\theta)=\frac1N\sum_t\rho_t(\theta)\hat A_t,\qquad
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)},$$

subject to

$$\bar D_{KL}(\theta)=\frac1N\sum_t
D_{KL}(\pi_{\theta_{old}}(\cdot\mid s_t)\|\pi_\theta(\cdot\mid s_t))\le\delta,$$

where $\delta=$ `max_kl`. For discrete policies, categorical KL sums over
all actions at each sampled state. For continuous policies, analytic diagonal
Gaussian KL sums over action dimensions. A shared invertible tanh and affine
transform preserves KL, so bounded policies use their underlying Gaussians.
Floating-point saturation and inverse-action clamping can introduce numerical
approximation in bounded-action log probabilities.

## Conjugate gradients and backtracking

At the collecting policy, let $g=\nabla L$ and
$H=\nabla^2\bar D_{KL}$. Solve $(H+\eta I)x=g$ approximately using
`cg_steps` conjugate-gradient iterations, with $\eta=$ `damping` and $I$ the
identity. `cg_tolerance` bounds the squared residual norm. Hessian-vector
products use second derivatives through PyTorch rather than a dense Hessian.

The proposed step is

$$\Delta\theta=\sqrt{\frac{2\delta}{x^T(H+\eta I)x}}x.$$

Try $\theta_{old}+\alpha\Delta\theta$, with
$\alpha=1,\beta,\beta^2,\ldots$ and $\beta=$ `backtrack_coefficient`.
Accept only finite candidates whose actual batch KL is within `max_kl`,
whose surrogate gain is positive, and whose gain is at least
`accept_ratio` times the predicted gain $\alpha g^T\Delta\theta$.
After `backtrack_steps` failed candidates, restore the old actor exactly.
Nonpositive/nonfinite curvature or predicted gain skips the actor update.
An exception during line search also restores the old actor before propagating.

The critic then takes `value_epochs` full-rollout Adam steps. Separate weights
ensure critic fitting cannot invalidate the accepted actor constraint.
The sampled mean-KL check is an approximation to the theoretical trust-region
procedure; it does not bound KL at every state or guarantee higher true return.

## Settings and diagnostics

Defaults are `n_steps=1024`, `gamma=0.99`, `gae_lambda=0.95`,
`normalize_advantage=True`, `max_kl=0.01`, `damping=0.1`, `cg_steps=10`,
`cg_tolerance=1e-10`, `backtrack_steps=10`, `backtrack_coefficient=0.5`,
`accept_ratio=0.1`, `value_epochs=10`, and `value_learning_rate=1e-3`.
`max_grad_norm=1` clips critic gradients only. `entropy_coefficient` must be
zero: the actor maximizes the advantage surrogate without an entropy bonus.

TRPO inherits ActorCritic's construction and checkpoint plumbing, including
its `learning_rate` field and policy-optimizer slot. That optimizer is never
stepped, so `learning_rate` has no effect on the TRPO actor. Its step size is
set by the KL budget and line search.

`train/kl` measures accepted batch KL, `train/surrogate_gain` its improvement,
and `train/step_fraction` the accepted fraction. All three are zero for skipped
or rejected steps. `train/value_loss` is the MSE before the last critic step;
`train/entropy` measures policy entropy after the update (pre-squash Gaussian
entropy for bounded actions).

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
from aprenderl import TRPO, TRPOConfig

env = gym.make("CartPole-v1")
try:
    agent = TRPO(env, config=TRPOConfig(n_steps=256, seed=7), device="cpu")
    agent.learn(20_000)
    agent.save("artifacts/trpo.pt")
finally:
    env.close()
```

## Continuous-action example

```python
import gymnasium as gym
from aprenderl import TRPO, TRPOConfig

env = gym.make("Pendulum-v1")
try:
    agent = TRPO(env, config=TRPOConfig(n_steps=1024, seed=7), device="cpu")
    agent.learn(51_200)
    observation, _ = env.reset(seed=42)
    action = agent.predict(observation, deterministic=True)
    assert env.action_space.contains(action)
    agent.save("artifacts/trpo_continuous.pt")
    restored = TRPO.load("artifacts/trpo_continuous.pt", env, device="cpu")
finally:
    env.close()
```

The default `GaussianPolicyNetwork` learns both means and log standard
deviations. Actions retain the environment's shape and floating dtype, including
multidimensional boxes; log densities sum over all flattened action dimensions.
Deterministic actions use the transformed mean, not a random sample.

The [continuous-action notebook](../../examples/trpo_continuous.ipynb)
uses lower-gravity Pendulum for a compact CPU demonstration, plots episode
returns, and renders deterministic evaluation in a separate environment.

See the [source](../../src/aprenderl/algorithms/trpo.py),
[public-API example](../../examples/trpo.ipynb), and
[from-scratch study notebook](../../study/06_policy_gradients/06_trpo.ipynb).

## Reference

John Schulman et al. (2015),
[*Trust Region Policy Optimization*](https://arxiv.org/abs/1502.05477).
