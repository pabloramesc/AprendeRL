# Discrete Soft Actor-Critic

Discrete SAC retains replay, twin critics, and temperature tuning from
[SAC](sac.md), but replaces continuous-action Monte Carlo expectations with
exact sums over a finite action set. It is an off-policy stochastic-policy
algorithm despite being listed at the end of the continuous-control learning
path.

## Learning rule

The actor outputs logits and probabilities $\pi_\theta(a\mid s)$ through
softmax. Each critic $Q_{\phi_i}(s)$ outputs a vector with one value per action.
$s,a,r,s'$ denote a recorded transition, $d$ true termination, $\gamma$ the
discount, $\theta$ actor weights, and $\phi_i$ critic weights. Bars indicate
target critics. With entropy temperature $\alpha$:

$$V_{\mathrm{target}}(s')=\sum_a\pi_\theta(a\mid s')
[\min_iQ_{\bar\phi_i}(s',a)-\alpha\log\pi_\theta(a\mid s')],$$

$$y=r+\gamma(1-d)V_{\mathrm{target}}(s'),\qquad
L_Q=\sum_i\mathbb E[(Q_{\phi_i}(s,a)-y)^2].$$

The critic loss gathers **recorded** action indices; its target is detached.
The actor instead sums over every current action:

$$L_\pi=\mathbb E_s\sum_a\pi_\theta(a\mid s)
[\alpha\log\pi_\theta(a\mid s)-\min_iQ_{\phi_i}(s,a)].$$

Detach critic values while preserving the actor probabilities and log
probabilities. No action reparameterization or tanh correction is required.
Use `log_softmax` to evaluate categorical log probabilities stably. Only the
critics have target copies, averaged after every gradient update:
$\bar\phi_i\leftarrow(1-\tau)\bar\phi_i+\tau\phi_i$.

## Temperature and exploration

For learned $\beta=\log\alpha$, use an exact entropy expectation:

$$L_\beta=-\mathbb E_s\left[\beta\operatorname{stopgrad}
\left(\sum_a\pi_\theta(a\mid s)\log\pi_\theta(a\mid s)+\mathcal H_*\right)\right].$$

Default `target_entropy` is $\mathcal H_*=0.98\log n$, with $n$ the action
count. Explicit targets must lie in $[0,\log n]$. This near-uniform target can
favor persistent exploration; tune it or use a fixed numeric `ent_coef` for
your task. The same automatic-temperature settings as SAC are available.

Warmup samples uniformly; subsequent behavior samples the categorical policy.
Deterministic prediction selects its mode. Replay stores zero-based action
indices, and prediction restores Gymnasium's `Discrete.start` offset.

## Public API

```python
import gymnasium as gym
from aprenderl import DiscreteSAC, DiscreteSACConfig
from aprenderl.utils import evaluate_policy

config = DiscreteSACConfig(batch_size=128, learning_starts=1_000, seed=7)
env = gym.make("CartPole-v1")
try:
    agent = DiscreteSAC(env, config=config, device="cpu")
    agent.learn(total_timesteps=20_000)
    agent.save("artifacts/discrete_sac.pt")
finally:
    env.close()

evaluation_env = gym.make("CartPole-v1")
try:
    restored = DiscreteSAC.load("artifacts/discrete_sac.pt", evaluation_env, device="cpu")
    result = evaluate_policy(restored, evaluation_env, episodes=5, seed=1_000)
    print(result.mean_return)
finally:
    evaluation_env.close()
```

`learn` supports `progress_bar`; construction supports `callback`, `logger`,
and `device`. Multiple `learn` calls continue counters and update cadence.
`predict(observation, deterministic=False)` handles one observation at a time.

| Setting | Meaning |
| --- | --- |
| `learning_rate` | Actor Adam learning rate |
| `critic_learning_rate` | Critic Adam learning rate |
| `buffer_size`, `batch_size` | Replay capacity and sampled minibatch size |
| `learning_starts` | Uniform-action warmup and earliest training step |
| `train_freq`, `gradient_steps` | Environment-step update interval and updates per interval |
| `gamma`, `tau` | Discount and target averaging coefficient |
| `max_grad_norm` | Separate actor and critic gradient-norm bound |

Training begins once a full minibatch exists, warmup permits it, and the step
matches `train_freq`. This is a compact single-environment implementation;
vector environments and dictionary observations are outside its interface.

## Spaces and custom networks

Observations must be a non-scalar `Box`. The default MLP flattens observations.
Actions must be `Discrete`, including nonzero starts. Supply `policy_network`
with output `(batch, action_dim)` logits and `critic_networks=[q1, q2]`, each
returning a `(batch, action_dim)` Q vector from observations.

Actor and critics must have disjoint parameters. Custom modules are validated
at construction and used in evaluation mode, including during optimization,
to avoid changing dropout or batch-normalization state; autograd remains
active. Target copies never receive gradients. Target parameter averaging
also copies non-parameter buffers.

## Termination and checkpoints

Both termination and truncation reset the training environment. Only
`terminated` suppresses bootstrapping; replay keeps the final next observation
from a time limit before the reset observation is obtained.

Checkpoints contain online and target weights, Adam state, configuration,
training counters, completed-episode history, and behavior-noise RNG state.
SAC variants also persist learned temperature and its optimizer. Loading
validates algorithm identity, observation shape, and action shape, bounds or
discrete offset. Custom-network checkpoints require the corresponding
`policy_network` and/or `critic_networks` architecture when loading.

Replay contents, environment state, and global PyTorch sampling RNG are not
saved. Training can resume with a fresh replay buffer and reset environment;
this is not a bit-for-bit continuation of the prior trajectory. Existing
algorithm checkpoint formats are unchanged.

## Further reading

- [Christodoulou](https://arxiv.org/abs/1910.07207)
- [Public API example](../../examples/discrete_sac.ipynb)
- [From-scratch study notebook](../../study/07_continuous_control/05_discrete_sac.ipynb)
- [Continuous-control chapter](../../study/07_continuous_control/README.md)
- [Algorithm source](../../src/aprenderl/algorithms/discrete_sac.py)
