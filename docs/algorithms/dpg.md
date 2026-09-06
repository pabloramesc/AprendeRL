# Deterministic Policy Gradient (DPG)

DPG is the foundation of [DDPG](ddpg.md) and [TD3](td3.md). AprendeRL teaches it
in a [standalone study notebook](../../study/07_continuous_control/01_dpg.ipynb),
rather than exposing a separate `DPG` class. The public learning path starts
at DDPG, where replay and target networks support neural continuous control.

## The gradient

For deterministic policy $\mu_\theta(s)$, the deterministic policy-gradient
theorem expresses the return gradient through the action-value derivative:

$$\nabla_\theta J(\theta)=\int\rho^\mu(s)
\nabla_\theta\mu_\theta(s)
\nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}\,ds.$$

$J$ is the discounted-return objective, $\theta$ actor parameters,
$\rho^\mu$ the discounted state-visitation density, and $Q^\mu$ the policy's
action-value function. The integral is over states $s$; $a$ denotes actions.
The theorem assumes differentiability and suitable regularity conditions.
Off-policy actor-critic implementations approximate this update using states
from an exploratory behavior policy and a learned critic $Q_\phi$.

Minimizing the sampled surrogate

$$L_\mu=-\frac1N\sum_{s\in B}Q_\phi(s,\mu_\theta(s))$$

applies this chain rule through autograd. $B$ is a batch of $N$ states and
$\phi$ critic weights. Hold critic weights fixed during the actor step while
retaining gradients with respect to its action input. Unlike stochastic
likelihood-ratio methods, this update needs no log-probability term.

## Educational implementation

The notebook uses a one-step tracking task: $s\sim U(-1,1)$,
$a\in[-1,1]$, and reward $r=-(a-0.6s)^2$. A tanh-linear actor and quadratic
critic learn online, with Gaussian exploration and no replay or target copies.
The exact solution $a^*(s)=0.6s$ and action gradient
$\partial Q/\partial a=-2(a-0.6s)$ make the learned gradient directly testable.
The tanh-linear actor approximates the ideal linear policy over the interval.

This isolates the gradient concept; it does not claim to reproduce every
algorithm or compatible function-approximation condition in the original
paper. The general TD target is shown, but every episode terminates after
one step, so its bootstrap vanishes. Later lessons handle long horizons and
bootstrap across time-limit truncation.

## Reading order

1. [DPG](../../study/07_continuous_control/01_dpg.ipynb): action derivatives.
2. [DDPG](ddpg.md): deep actor/critic, replay and target networks.
3. [TD3](td3.md): twin critics, smoothing and delayed updates.
4. [SAC](sac.md): reparameterization and entropy tuning.
5. [Discrete SAC](discrete_sac.md): exact categorical expectations.

Reference: [Silver et al., 2014](https://proceedings.mlr.press/v32/silver14.html).
