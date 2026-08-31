# Deep Q-Network (DQN)

[Back to the algorithm index](../algorithms.md)

**Policy classification:** Off-policy. DQN can learn from replayed transitions
collected by older epsilon-greedy behavior policies while optimizing a greedy
target policy.

**Use when:** observations are numeric `Box` vectors or tensors and actions are
discrete, but a state-action table is impractical.

## Network and replay

DQN replaces the table with a neural network $Q_\theta(s,a)$. AprendeRL's
default network flattens the observation, passes it through two 128-unit ReLU
layers, and outputs one value per action.

Each transition is stored in a fixed-size circular replay buffer. Once training
starts, a batch of $B$ transitions is sampled uniformly with replacement. A
separate target network with parameters $\theta^-$ constructs targets:

$$
y_i = r_i + \gamma(1-d_i)
\max_{a'}Q_{\theta^-}(s'_i,a'),
$$

where $d_i=1$ only for a true Gymnasium `terminated` transition. Time-limit
truncations continue to bootstrap because the underlying Markov decision
process did not terminate.

The online prediction for the recorded action is

$$
q_i = Q_\theta(s_i,a_i).
$$

## Loss and target updates

The network minimizes mean Huber loss:

$$
\mathcal{L}(\theta)=\frac{1}{B}\sum_{i=1}^{B}h(q_i-y_i),
$$

$$
h(x)=
\begin{cases}
\frac{1}{2}x^2, & |x|\leq 1,\\
|x|-\frac{1}{2}, & |x|>1.
\end{cases}
$$

Targets are treated as constants during backpropagation. Adam updates $\theta$,
and the gradient norm is clipped to `max_grad_norm`. Every
`target_update_interval` environment steps, the target network is replaced by
a hard copy:

$$
\theta^- \leftarrow \theta.
$$

Replay reduces correlation between consecutive samples; the delayed target
network makes the regression target change more slowly. Training begins after
`learning_starts`, runs every `train_freq` environment steps, and performs
`gradient_steps` optimizer updates each time.

## Exploration and variants

Training uses linearly annealed epsilon-greedy exploration, while deterministic
prediction always chooses the action with the largest Q-value. See the
[shared notation](../algorithms.md#epsilon-greedy-exploration) for its exact
definition.

This class is **vanilla DQN**: the target network both selects and evaluates
$\arg\max_{a'}Q_{\theta^-}(s',a')$. AprendeRL also provides standalone Double,
Dueling, Prioritized, n-step, NoisyNet, C51, QR-DQN, IQN, and Rainbow classes.
See the [DQN variant guide](dqn_variants.md) for their targets and losses.

The implementation is in
[`src/aprenderl/algorithms/dqn.py`](../../src/aprenderl/algorithms/dqn.py).
