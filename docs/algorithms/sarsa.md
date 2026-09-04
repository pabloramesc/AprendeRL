# Tabular SARSA

[Back to the algorithm index](../algorithms.md)

**Policy classification:** On-policy. The epsilon-greedy action used in the
temporal-difference target is also used for the next environment interaction.

**Use when:** observations and actions are finite `Discrete` spaces small enough
to store every state-action pair, and you want the learned values to account
for the behavior policy's exploration.

## Update rule

The algorithm stores one value per pair in a table
$Q \in \mathbb{R}^{|\mathcal{S}|\times|\mathcal{A}|}$. After taking $a_t$ in
$s_t$, it samples the next epsilon-greedy action $a_{t+1}$ and computes

$$
\delta_t = r_{t+1} + \gamma(1-d_t)Q(s_{t+1},a_{t+1}) - Q(s_t,a_t),
$$

where $d_t=1$ only for a true Gymnasium `terminated` transition. It then
immediately updates one table entry:

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\delta_t,
\qquad 0 < \alpha \leq 1.
$$

The sampled $a_{t+1}$ is retained and used for the next interaction. This is
the implementation detail that makes SARSA on-policy instead of a target that
merely resembles SARSA.

A true terminal state has no future value or next action. A time-limit
`truncated` transition still samples a behavior action and bootstraps from the
final observation because the underlying Markov decision process did not
terminate.

## Exploration

Training uses the same linearly decayed epsilon-greedy behavior policy as
Q-Learning:

$$
a_t =
\begin{cases}
\text{uniform random action}, & u < \varepsilon_t,\\
\arg\max_a Q(s_t,a), & \text{otherwise},
\end{cases}
\qquad u \sim \mathcal{U}(0,1).
$$

Deterministic prediction always uses the greedy action.

## SARSA versus Q-Learning

Q-Learning bootstraps from $\max_{a'}Q(s_{t+1},a')$, regardless of which action
the exploratory behavior policy will take. SARSA instead bootstraps from
$Q(s_{t+1},a_{t+1})$. It therefore learns the value of its current exploratory
policy and can favor safer behavior when exploration itself carries risk.

The classical convergence result requires a finite MDP, sufficient exploration,
and suitably decreasing per-pair learning rates. AprendeRL currently uses the
configured constant $\alpha$, which is practical but does not satisfy that
exact guarantee.

AprendeRL uses a contiguous `float32` NumPy array for constant-time access and
low per-value memory overhead. `initial_q_value` fills the table before the
first update; a positive value can produce optimistic initial exploration.

The implementation is in
[`src/aprenderl/algorithms/sarsa.py`](../../src/aprenderl/algorithms/sarsa.py).

## Original report

- Gavin A. Rummery and Mahesan Niranjan (1994),
  [*On-line Q-learning Using Connectionist Systems*](https://www.cs.utexas.edu/~shivaram/readings/b2hd-RummeryNiranjan1994.html),
  Cambridge University Engineering Department, Technical Report
  CUED/F-INFENG/TR 166. The report introduced the update later named SARSA.
