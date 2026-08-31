# DQN variants and distributional value learning

[Back to the algorithm index](../algorithms.md)

Every class in this guide preserves the DQN public interface and accepts a
numeric `Box` observation space with a `Discrete` action space. The variants
reuse DQN's target-update schedule, optimizer, termination mask, exploration
schedule where applicable, callbacks, logging, and checkpoint format.

## Double DQN

Vanilla DQN uses the target network to select and evaluate the next action.
`DoubleDQN` reduces maximization bias by selecting online and evaluating with
the target network:

$$
y=R+\gamma(1-d)
Q_{\theta^-}\left(S',\arg\max_aQ_\theta(S',a)\right).
$$

## Dueling DQN

`DuelingDQN` replaces the default network with value and advantage streams:

$$
Q(s,a)=V(s)+A(s,a)-\frac{1}{|\mathcal A|}\sum_{a'}A(s,a').
$$

Centering the advantages makes the decomposition identifiable while retaining
one Q-value per action at the public boundary.

## Prioritized experience replay

`PrioritizedDQN` samples transition $i$ in proportion to its absolute TD error:

$$
P(i)=\frac{p_i^\alpha}{\sum_kp_k^\alpha},\qquad
w_i=(N P(i))^{-\beta}.
$$

Weights are normalized within a batch and multiply the element-wise Huber
loss. `priority_beta` anneals to one over `priority_beta_steps` interactions.
The compact NumPy implementation uses explicit arrays so the sampling rule is
easy to inspect.

## n-step DQN

`NStepDQN` stores an accumulated reward and exact bootstrap discount:

$$
y=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+\gamma^n(1-d)\max_aQ_{\theta^-}(S_{t+n},a).
$$

Short returns at episode boundaries retain their actual exponent. This is
particularly important for Gymnasium truncations, which still bootstrap.

## NoisyNet DQN

`NoisyDQN` replaces epsilon-greedy exploration with factorized Gaussian noisy
linear layers:

$$
W=\mu_W+\sigma_W\odot\varepsilon_W.
$$

Noise is resampled for behavior decisions and gradient updates. Deterministic
prediction switches the network to evaluation mode and uses the learned mean
parameters only.

## C51

`C51` represents the return distribution with probabilities on a fixed support
of `atoms` values between `v_min` and `v_max`. The Bellman-updated atoms are
clipped and linearly projected back to this support. Training minimizes the
cross-entropy

$$
\mathcal L=-\sum_i(\Phi T Z)_i\log p_i(s,a).
$$

Actions use the expectation of the categorical distribution.

## QR-DQN

`QRDQN` predicts `quantiles` equally weighted return locations rather than
fixed categorical probabilities. Given predicted quantile $\theta_i$ and
target quantile $y_j$, it minimizes the pairwise quantile Huber loss

$$
\sum_{i,j}\left|\tau_i-\mathbf 1\{y_j-\theta_i<0\}\right|
L_\kappa(y_j-\theta_i).
$$

The mean of the predicted quantiles determines the action value.

## IQN

`IQN` samples quantile fractions $\tau\sim U(0,1)$ for each update. A cosine
embedding of $\tau$ is multiplied with the state features before the action
head, approximating the full inverse return distribution rather than a fixed
quantile grid. Deterministic prediction uses evenly spaced midpoint fractions
to avoid sampling noise.

## Rainbow DQN

`RainbowDQN` composes the six canonical improvements in one implementation:

- Double action selection;
- dueling categorical heads;
- proportional prioritized replay;
- n-step returns;
- NoisyNet exploration;
- the C51 distributional target.

The components remain visible as ordinary methods and modules rather than
being hidden behind a generic feature-flag system.

## Minimal API

```python
import gymnasium as gym

from aprenderl import RainbowDQN, RainbowDQNConfig

env = gym.make("CartPole-v1")
config = RainbowDQNConfig(buffer_size=20_000, learning_starts=1_000)
agent = RainbowDQN(env, config=config).learn(20_000)

agent.save("rainbow.pt")
restored = RainbowDQN.load("rainbow.pt", env)
```

The algorithms are implemented in
[`distributional_dqn.py`](../../src/aprenderl/algorithms/distributional_dqn.py)
and [`dqn_variants.py`](../../src/aprenderl/algorithms/dqn_variants.py). The
network components are in
[`value_based.py`](../../src/aprenderl/networks/value_based.py).
