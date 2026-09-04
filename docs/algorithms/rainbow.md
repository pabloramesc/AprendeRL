# Rainbow DQN

[Back to the algorithm index](../algorithms.md)

**Policy classification:** Off-policy. Rainbow learns from prioritized replay
while its NoisyNet behavior policy changes as the learned noise scales change.

**Use when:** observations are numeric `Box` vectors or tensors, actions are
`Discrete`, and the complete paper-defined combination is preferable to a
single DQN improvement. This implementation intentionally does not expose a
feature-flag matrix for disabling individual Rainbow components.

## The six-part combination

`RainbowDQN` combines six mechanisms in one agent:

1. Double DQN selects the next action with the online network and evaluates its
   distribution with the target network.
2. Dueling heads separately estimate categorical state values and advantages.
3. Proportional prioritized replay samples more informative stored returns.
4. n-step returns move reward information across several environment steps.
5. factorized NoisyNet layers provide learned exploration without epsilon.
6. C51 represents returns as probabilities on a fixed categorical support.

The class inherits the environment loop, training schedule, target-network
updates, checkpoint format, and gradient clipping from `DQN`, but owns the
network, replay, action-selection, target, and loss behavior required by this
combination.

## Dueling categorical NoisyNet

The default `CategoricalQNetwork` uses noisy linear layers in both its feature
extractor and output heads. Its dueling logits are

$$
\ell(s,a,z_i)=V(s,z_i)+A(s,a,z_i)
-\frac{1}{|\mathcal A|}\sum_{a'}A(s,a',z_i).
$$

A softmax over atoms produces $p_\theta(z_i\mid s,a)$, and actions maximize the
distributional expectation $\sum_i z_i p_\theta(z_i\mid s,a)$. During training
and non-deterministic prediction, fresh factorized Gaussian parameter noise is
sampled. `predict(..., deterministic=True)` temporarily uses evaluation mode,
so every noisy layer uses its learned mean parameters.

Rainbow's inherited epsilon schedule is replaced with a constant zero schedule;
`exploration_initial_epsilon`, `exploration_final_epsilon`, and
`exploration_steps` therefore do not control its actions.

## n-step categorical target

For a horizon $h\leq n$, the replay item stores

$$
R_t^{(h)}=\sum_{k=0}^{h-1}\gamma^k r_{t+k+1}
\quad\text{and}\quad
\Gamma_h=\gamma^h.
$$

Full n-step returns are stored during an episode. At either a terminated or
truncated episode boundary, the pending queue is flushed as progressively
shorter returns. The terminal flag remains separate: a true termination removes
the bootstrap, while a time-limit truncation retains it at the final
observation.

The online network selects

$$
a^*=\arg\max_a\sum_i z_i p_\theta(z_i\mid s_{t+h},a),
$$

and the target network supplies the categorical probabilities for that action.
Its atoms become

$$
\tilde z_i=R_t^{(h)}+\Gamma_h(1-d)z_i.
$$

As in C51, the shifted distribution is clipped and linearly projected onto the
fixed support from `v_min` to `v_max`.

## Prioritized loss

New replay items receive the current maximum priority. Existing item $i$ is
sampled according to

$$
P(i)=\frac{p_i^\alpha}{\sum_k p_k^\alpha}.
$$

The importance-sampling correction is normalized by the largest weight in the
batch:

$$
w_i=\frac{(N P(i))^{-\beta}}{\max_j (N P(j))^{-\beta}}.
$$

`priority_beta` increases linearly from its configured value to one over
`priority_beta_steps` environment interactions. If $\ell_i$ is the per-item C51
cross-entropy, Rainbow minimizes

$$
\mathcal L=\frac{1}{B}\sum_i w_i\ell_i
$$

and then sets $p_i\leftarrow\ell_i+\varepsilon_p$. The logger reports the
weighted loss, gradient norm, mean updated priority, update count, and a fixed
epsilon of zero.

## Usage

```python
import gymnasium as gym

from aprenderl import RainbowDQN, RainbowDQNConfig

env = gym.make("CartPole-v1")
config = RainbowDQNConfig(
    atoms=51,
    v_min=-10.0,
    v_max=10.0,
    n_steps=3,
    priority_alpha=0.5,
    priority_beta=0.4,
    noise_sigma=0.5,
    buffer_size=20_000,
    learning_starts=1_000,
)
agent = RainbowDQN(env, config=config).learn(20_000)

agent.save("rainbow.pt")
restored = RainbowDQN.load("rainbow.pt", env)
```

`double_dqn` must remain `True`. A custom `network` must return logits shaped
`(batch_size, action_count, atoms)`. To preserve Rainbow behavior, it should
also contain resettable `NoisyLinear` layers and implement the dueling
decomposition; shape validation alone cannot establish those architectural
properties. Loading a custom-network checkpoint requires supplying the same
network architecture again.

The agent and projection helper are in
[`src/aprenderl/algorithms/rainbow.py`](../../src/aprenderl/algorithms/rainbow.py).
Its categorical and noisy network components are in
[`src/aprenderl/networks/distributional.py`](../../src/aprenderl/networks/distributional.py)
and [`src/aprenderl/networks/q_network.py`](../../src/aprenderl/networks/q_network.py),
and prioritized n-step storage is in
[`src/aprenderl/buffers/replay.py`](../../src/aprenderl/buffers/replay.py).
See the runnable [`Rainbow example`](../../examples/train_rainbow_dqn.ipynb)
and the standalone [`Rainbow study notebook`](../../study/rainbow_dqn.ipynb).

## Original and component papers

- Matteo Hessel et al. (2018),
  [*Rainbow: Combining Improvements in Deep Reinforcement Learning*](https://doi.org/10.1609/aaai.v32i1.11796),
  AAAI-18.
- Hado van Hasselt, Arthur Guez, and David Silver (2016),
  [*Deep Reinforcement Learning with Double Q-Learning*](https://doi.org/10.1609/aaai.v30i1.10295),
  AAAI-16.
- Ziyu Wang et al. (2016),
  [*Dueling Network Architectures for Deep Reinforcement Learning*](https://proceedings.mlr.press/v48/wangf16.html),
  ICML 2016.
- Tom Schaul et al. (2016),
  [*Prioritized Experience Replay*](https://arxiv.org/abs/1511.05952),
  ICLR 2016.
- Meire Fortunato et al. (2018),
  [*Noisy Networks for Exploration*](https://arxiv.org/abs/1706.10295),
  ICLR 2018.
- Marc G. Bellemare, Will Dabney, and Rémi Munos (2017),
  [*A Distributional Perspective on Reinforcement Learning*](https://proceedings.mlr.press/v70/bellemare17a.html),
  ICML 2017.
