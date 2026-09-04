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

## Exploration and Double DQN

Training uses linearly annealed epsilon-greedy exploration, while deterministic
prediction always chooses the action with the largest Q-value. See the
[shared notation](../algorithms.md#epsilon-greedy-exploration) for its exact
definition.

With the default `double_dqn=False`, the target network both selects and
evaluates $\arg\max_{a'}Q_{\theta^-}(s',a')$, matching vanilla DQN. Enabling
`double_dqn=True` separates selection and evaluation:

$$
y_i = r_i + \gamma(1-d_i)
Q_{\theta^-}\!\left(s'_i,\arg\max_{a'}Q_\theta(s'_i,a')\right).
$$

This is Double DQN without a second public class:

```python
config = DQNConfig(double_dqn=True)
agent = DQN(env, config=config)
```

The maintained distributional agents build on this training loop but replace
the scalar TD loss with different return-distribution models. Their separate
guides cover [C51](c51.md), [Rainbow DQN](rainbow.md),
[QR-DQN](qr_dqn.md), [IQN](iqn.md), and [FQF](fqf.md).

## Usage

```python
import gymnasium as gym

from aprenderl import DQN, DQNConfig

env = gym.make("CartPole-v1")
config = DQNConfig(
    buffer_size=20_000,
    learning_starts=1_000,
    double_dqn=True,
)
agent = DQN(env, config=config).learn(20_000)

action = agent.predict(env.reset()[0], deterministic=True)
agent.save("dqn.pt")
restored = DQN.load("dqn.pt", env)
```

A custom `network` must accept an observation batch and return one scalar per
action with shape `(batch_size, action_count)`. Checkpoints include the online
and target networks, optimizer, configuration, counters, episode history, and
exploration generator state. Loading a custom-network checkpoint requires the
same architecture through `network=...`.

The implementation is in
[`src/aprenderl/algorithms/dqn.py`](../../src/aprenderl/algorithms/dqn.py), the
default network is in
[`src/aprenderl/networks/q_network.py`](../../src/aprenderl/networks/q_network.py),
and uniform replay is in
[`src/aprenderl/buffers/replay.py`](../../src/aprenderl/buffers/replay.py).
See the runnable [`DQN example`](../../examples/train_dqn.ipynb) and the
standalone [`DQN study notebook`](../../study/dqn.ipynb).

## Original papers

- Volodymyr Mnih et al. (2015),
  [*Human-level control through deep reinforcement learning*](https://doi.org/10.1038/nature14236),
  Nature 518, 529–533.
- Hado van Hasselt, Arthur Guez, and David Silver (2016),
  [*Deep Reinforcement Learning with Double Q-Learning*](https://doi.org/10.1609/aaai.v30i1.10295),
  AAAI-16.
