# REINFORCE

[Back to the algorithm index](../algorithms.md)

**Policy classification:** On-policy. Every update uses complete trajectories
sampled from the current categorical policy, before that policy changes.

**Use when:** observations are numeric `Box` vectors or tensors and actions are
discrete, and a compact, educational on-policy baseline is more important than
sample efficiency.

## Policy and returns

REINFORCE parameterizes a categorical policy $\pi_\theta(a\mid s)$. The default
network maps each observation to one logit per action; sampling from the
resulting categorical distribution provides exploration without a separate
epsilon schedule.

For each completed episode, AprendeRL computes discounted reward-to-go:

$$
G_t = \sum_{k=t}^{T-1}\gamma^{k-t}r_{k+1}.
$$

## Policy-gradient loss

The algorithm minimizes the negative Monte Carlo policy-gradient objective over
all $N$ transitions collected from `episodes_per_update` episodes:

$$
\mathcal{L}_{\mathrm{policy}}(\theta)
= -\frac{1}{N}\sum_{t=0}^{N-1}
G_t\log\pi_\theta(a_t\mid s_t).
$$

By default, returns are standardized within each update batch. This does not
change their ordering and usually reduces gradient variance. An optional
entropy bonus encourages broader action distributions:

$$
\mathcal{L}(\theta)=\mathcal{L}_{\mathrm{policy}}(\theta)
-\beta\frac{1}{N}\sum_t\mathcal{H}\left(\pi_\theta(\cdot\mid s_t)\right).
$$

Adam updates the policy and the gradient norm is clipped to `max_grad_norm`.
Because vanilla REINFORCE has no learned value function, it waits for a full
episode and does not bootstrap at time-limit truncations. This makes the method
simple but generally more variable and less sample-efficient than actor-critic
methods.

The implementation is in
[`src/aprenderl/algorithms/reinforce.py`](../../src/aprenderl/algorithms/reinforce.py),
and the runnable example is
[`examples/train_reinforce.ipynb`](../../examples/train_reinforce.ipynb).
