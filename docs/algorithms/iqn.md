# Implicit Quantile Network (IQN)

[Back to the algorithm index](../algorithms.md)

**Policy classification:** Off-policy. IQN learns from uniform replay while an
epsilon-greedy behavior policy collects transitions.

**Use when:** observations are numeric `Box` vectors or tensors, actions are
`Discrete`, and the return distribution should be modeled as a continuous
quantile function instead of a fixed set of categorical atoms or quantile
fractions.

## Conditioning on quantile fractions

IQN approximates the inverse return distribution $Z_\tau(s,a)$ for arbitrary
$\tau\in[0,1]$. During training, AprendeRL samples `quantiles` independent
fractions for every observation:

$$
\tau_i\sim\mathcal U(0,1).
$$

For `embedding_dim` $D$, each fraction becomes a cosine feature vector

$$
c(\tau)=\left[
\cos(\pi\tau),\cos(2\pi\tau),\ldots,\cos(D\pi\tau)
\right].
$$

A learned linear layer and ReLU map this vector to the state-feature width. The
result is multiplied elementwise with the flattened observation's feature
embedding, and a final head produces one quantile value per action:

$$
Z_\tau(s,a)=g\!\left(f(s)\odot\phi(\tau),a\right).
$$

The default `ImplicitQuantileNetwork` uses two 128-unit ReLU layers for $f$ and
64 cosine features for $\phi$.

## Training and action selection

For each replay batch, the online network predicts $N=$ `quantiles` values at
freshly sampled fractions. The target network independently produces
$N'=$ `target_quantiles` samples for the selected next action:

$$
y_j=r+\gamma(1-d)Z^-_{\tau'_j}(s',a^*),
\qquad \tau'_j\sim\mathcal U(0,1).
$$

The target network chooses $a^*$ by default; with `double_dqn=True`, the online
network chooses it. In either case, AprendeRL estimates action values for this
choice using an evenly spaced midpoint grid rather than random fractions:

$$
\bar\tau_i=\frac{i+\tfrac12}{N},
\qquad
Q(s,a)\approx\frac1N\sum_i Z_{\bar\tau_i}(s,a).
$$

The same deterministic midpoint integration is used by epsilon-greedy behavior
and by `predict`, so action values do not vary merely because new fractions were
sampled. A non-deterministic prediction can still choose a random action through
epsilon. A true `terminated` transition removes the target bootstrap;
`truncated` does not.

## Quantile loss

IQN uses the pairwise quantile Huber loss described in the
[QR-DQN guide](qr_dqn.md#quantile-huber-loss), but its weights are the sampled
$\tau_i$ rather than a fixed grid:

$$
\mathcal L_{\mathrm{IQN}}
=\frac{1}{BNN'}\sum_b\sum_i\sum_j
\left|\tau_{b,i}-\mathbf 1\{y_{b,j}-Z_{\tau_{b,i}}(s_b,a_b)<0\}\right|
\frac{H_\kappa(y_{b,j}-Z_{\tau_{b,i}}(s_b,a_b))}{\kappa}.
$$

The remaining replay schedule, target copies, Adam update, and gradient clipping
are inherited from `DQN`.

## Usage

```python
import gymnasium as gym

from aprenderl import IQN, IQNConfig

env = gym.make("CartPole-v1")
config = IQNConfig(
    quantiles=32,
    target_quantiles=32,
    embedding_dim=64,
    buffer_size=20_000,
    learning_starts=1_000,
)
agent = IQN(env, config=config).learn(20_000)

agent.save("iqn.pt")
restored = IQN.load("iqn.pt", env)
```

IQN's custom-network contract is richer than a plain output shape. Calling
`network(observations, num_quantiles)` must return `(values, taus)` with shapes
`(batch_size, action_count, num_quantiles)` and
`(batch_size, num_quantiles)`. The network must also implement
`q_values(observations, num_quantiles)` and return
`(batch_size, action_count)`. A custom-network checkpoint requires the same
architecture at load time.

The algorithm and loss helper are in
[`src/aprenderl/algorithms/iqn.py`](../../src/aprenderl/algorithms/iqn.py), and
the default network is in
[`src/aprenderl/networks/distributional.py`](../../src/aprenderl/networks/distributional.py).
See the runnable [`IQN example`](../../examples/train_iqn.ipynb) and the
standalone [`IQN study notebook`](../../study/iqn.ipynb).

## Original paper

- Will Dabney, Georg Ostrovski, David Silver, and Rémi Munos (2018),
  [*Implicit Quantile Networks for Distributional Reinforcement Learning*](https://proceedings.mlr.press/v80/dabney18a.html),
  ICML 2018.
