# Tabular Q-Learning

[Back to the algorithm index](../algorithms.md)

**Policy classification:** Off-policy. The epsilon-greedy behavior policy can
differ from the greedy policy used in the temporal-difference target.

**Use when:** observations and actions are finite `Discrete` spaces small enough
to store every state-action pair.

## Update rule

The algorithm stores one value per pair in a table
$Q \in \mathbb{R}^{|\mathcal{S}|\times|\mathcal{A}|}$. For each observed
transition, it computes

$$
\delta_t = r_{t+1} + \gamma(1-d_t)\max_{a'}Q(s_{t+1},a') - Q(s_t,a_t),
$$

where $d_t=1$ only for a true Gymnasium `terminated` transition. It then
immediately updates one table entry:

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\delta_t,
\qquad 0 < \alpha \leq 1.
$$

A true terminal state has no future value. A time-limit `truncated` transition
still bootstraps because the underlying Markov decision process did not
terminate.

## Exploration

Training uses epsilon-greedy actions:

$$
a_t =
\begin{cases}
\text{uniform random action}, & u < \varepsilon_t,\\
\arg\max_a Q(s_t,a), & \text{otherwise},
\end{cases}
\qquad u \sim \mathcal{U}(0,1).
$$

Epsilon decreases linearly for $T$ environment steps, then remains at its final
value:

$$
\varepsilon_t = \varepsilon_{\mathrm{start}} +
\min\left(\frac{t}{T},1\right)
\left(\varepsilon_{\mathrm{final}}-\varepsilon_{\mathrm{start}}\right).
$$

Deterministic prediction always uses the greedy action.

## Implementation notes

$\alpha$ is the learning rate. The `max` target is greedy even when the
behavior action was exploratory, so Q-Learning is **off-policy**. The classical
convergence result requires a finite MDP, sufficient exploration, and suitably
decreasing per-pair learning rates. AprendeRL currently uses the configured
constant $\alpha$, which is practical but does not satisfy that exact guarantee.

AprendeRL uses a contiguous `float32` NumPy array for constant-time access and
low per-value memory overhead. `initial_q_value` fills the table before the
first update; a positive value can produce optimistic initial exploration.

The implementation is in
[`src/aprenderl/algorithms/q_learning.py`](../../src/aprenderl/algorithms/q_learning.py).
