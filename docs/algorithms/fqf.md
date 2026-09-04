# Fully Parameterized Quantile Function (FQF)

[Back to the algorithm index](../algorithms.md)

**Policy classification:** Off-policy. FQF trains from uniformly replayed
transitions collected by epsilon-greedy behavior.

**Use when:** observations are numeric `Box` vectors or tensors, actions are
`Discrete`, and both the return quantiles and where they are evaluated should be
learned. FQF is the most intricate maintained distributional DQN because it
optimizes these two parts separately.

## Learned fraction intervals

QR-DQN fixes quantile fractions on a uniform grid, while IQN samples them. FQF
uses a fraction proposal network to learn state-dependent interval widths. For
`quantiles` $N$, a softmax produces $q_i>0$ with $\sum_i q_i=1$. Their cumulative
sums define $N+1$ boundaries and $N$ midpoints:

$$
\tau_0=0,\qquad
\tau_i=\sum_{j=0}^{i-1}q_j,\qquad
\hat\tau_i=\frac{\tau_i+\tau_{i+1}}{2}.
$$

The quantile-value path evaluates $Z_{\hat\tau_i}(s,a)$ using the same cosine
embedding construction as IQN. Expected action values weight each midpoint by
its learned interval width:

$$
Q(s,a)=\sum_{i=0}^{N-1}
(\tau_{i+1}-\tau_i)Z_{\hat\tau_i}(s,a).
$$

The default `FQFNetwork` uses a two-layer, 128-unit state feature extractor, a
linear fraction proposal head, 64 cosine features, and a linear action-value
head. State embeddings are computed once per batch. They are detached before
the fraction proposal step, so its optimizer does not modify the shared feature
extractor.

## Fraction proposal update

For each internal boundary $\tau_i$, the network evaluates its return value and
compares it with the two neighboring midpoint values. AprendeRL implements the
paper's fraction gradient using

$$
g_i=2Z_{\tau_i}(s,a)
-Z_{\hat\tau_{i-1}}(s,a)-Z_{\hat\tau_i}(s,a),
\qquad i=1,\ldots,N-1.
$$

The values in $g_i$ are detached, leaving the boundaries as the differentiated
variables. The fraction objective is

$$
\mathcal L_{\mathrm{fraction}}
=\frac1B\sum_b\sum_{i=1}^{N-1}
\operatorname{stopgrad}(g_{b,i})\tau_{b,i}
-\beta\mathcal H(q_b),
$$

where `entropy_coefficient` is $\beta$. The entropy bonus can discourage the
proposal distribution from collapsing too early. An RMSprop optimizer updates
only the fraction head at `fraction_learning_rate`; its gradient norm is clipped
to `max_grad_norm`.

## Quantile-value update

The next action maximizes the target network's expected value by default, or the
online network's value with `double_dqn=True`. The target network evaluates the
next observation at the detached midpoint fractions generated for the current
batch:

$$
y_i=r+\gamma(1-d)Z^-_{\hat\tau_i}(s',a^*).
$$

Only true `terminated` transitions set $d=1$; a Gymnasium `truncated` transition
still bootstraps. The predicted and target values use the pairwise quantile
Huber loss from the [QR-DQN guide](qr_dqn.md#quantile-huber-loss), weighted by
the learned midpoint fractions $\hat\tau_i$.

A separate Adam optimizer updates the state encoder, cosine embedding, and
quantile-value head at `learning_rate`. The fraction and quantile optimizer
parameter sets are disjoint, and both optimizer states are included in a saved
checkpoint. Hard target-network updates and replay scheduling remain inherited
from `DQN`.

## Usage

```python
import gymnasium as gym

from aprenderl import FQF, FQFConfig

env = gym.make("CartPole-v1")
config = FQFConfig(
    quantiles=32,
    embedding_dim=64,
    fraction_learning_rate=2.5e-9,
    entropy_coefficient=0.0,
    buffer_size=20_000,
    learning_starts=1_000,
)
agent = FQF(env, config=config).learn(20_000)

agent.save("fqf.pt")
restored = FQF.load("fqf.pt", env)
```

A custom FQF network must implement the complete interface used by the two
updates: `state_embeddings`, `fractions`, `quantile_values`, `q_values`,
`fraction_parameters`, and `quantile_parameters`. Its forward result must be
`(values, taus, tau_hats, entropy)` with shapes
`(batch_size, action_count, quantiles)`,
`(batch_size, quantiles + 1)`, `(batch_size, quantiles)`, and `(batch_size,)`.
Loading a custom-network checkpoint requires supplying the same architecture.

The algorithm and both loss helpers are in
[`src/aprenderl/algorithms/fqf.py`](../../src/aprenderl/algorithms/fqf.py), and
the default network is in
[`src/aprenderl/networks/distributional.py`](../../src/aprenderl/networks/distributional.py).
See the runnable [`FQF example`](../../examples/fqf.ipynb).

## Original paper

- Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, and Tie-Yan Liu
  (2019),
  [*Fully Parameterized Quantile Function for Distributional Reinforcement Learning*](https://proceedings.neurips.cc/paper/2019/hash/f471223d1a1614b58a7dc45c9d01df19-Abstract.html),
  NeurIPS 2019.
