# Deep Deterministic Policy Gradient (DDPG)

DDPG learns a deterministic actor $\mu_\theta(s)$ and a critic $Q_\phi(s,a)$
from replay. It adds deep function approximation, replay, and slowly moving
target networks to the [DPG learning rule](dpg.md).

## Learning rule

Sample $(s,a,r,s',d)$ from replay, where $s,s'$ are observations, $a$ is the
recorded behavior action, $r$ the reward, and $d$ is one for true termination.
The discount is $\gamma$, actor weights are $\theta$, and critic weights are
$\phi$. Bars denote target-network weights.

$$
y=r+\gamma(1-d)Q_{\bar\phi}(s',\mu_{\bar\theta}(s')),
\qquad L_Q=\mathbb E[(Q_\phi(s,a)-y)^2].
$$

Fit the critic to the detached target, then improve the actor:

$$
L_\mu=-\mathbb E[Q_\phi(s,\mu_\theta(s))],\qquad
\nabla_\theta L_\mu=-\mathbb E[
\nabla_aQ_\phi(s,a)|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)].
$$

Freeze critic parameters during actor backpropagation, preserving derivatives
with respect to the action. Both targets move after every gradient step:

$$\bar w\leftarrow(1-\tau)\bar w+\tau w.$$

Here $w$ denotes online parameters and $\tau$ is `tau`. The implementation
uses MSE, Adam, and separate actor/critic gradient-norm clipping.

## Exploration and bounds

The network outputs normalized actions $u\in[-1,1]$. For bounds $l,h$, the
environment action is $a=(h+l)/2+(h-l)u/2$. Before `learning_starts`, behavior
uses uniform action-space samples. Subsequently, independent Gaussian noise
with standard deviation `action_noise` is added in normalized coordinates,
then clipped and rescaled. Deterministic prediction omits noise even during
warmup. This compact implementation uses Gaussian noise rather than the
original paper's temporally correlated Ornstein–Uhlenbeck process; it does
not include batch normalization by default.

## Public API

```python
import gymnasium as gym
from aprenderl import DDPG, DDPGConfig
from aprenderl.utils import evaluate_policy

config = DDPGConfig(batch_size=128, learning_starts=1_000, seed=7)
env = gym.make("Pendulum-v1")
try:
    agent = DDPG(env, config=config, device="cpu")
    agent.learn(total_timesteps=20_000)
    agent.save("artifacts/ddpg.pt")
finally:
    env.close()

evaluation_env = gym.make("Pendulum-v1")
try:
    restored = DDPG.load("artifacts/ddpg.pt", evaluation_env, device="cpu")
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
Actions must be a non-scalar floating-point `Box` with finite, strictly
ordered bounds. Multidimensional action shapes and asymmetric per-dimension
bounds are preserved by prediction; replay flattens actions for critics.
Unbounded or partially bounded action spaces are rejected.
Supply `policy_network` returning normalized `(batch, action_dim)` actions
in `[-1, 1]`; the algorithm performs environment rescaling. The default is
`DeterministicPolicyNetwork`.
Supply `critic_networks` as a sequence of 1 independent module.
Each accepts `(observations, actions)` and returns `(batch, 1)`;
actions have shape `(batch, action_dim)` in environment units. The default
is `ContinuousQNetwork`.

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

- [Lillicrap et al.](https://arxiv.org/abs/1509.02971)
- [Public API example](../../examples/ddpg.ipynb)
- [From-scratch study notebook](../../study/07_continuous_control/02_ddpg.ipynb)
- [Continuous-control chapter](../../study/07_continuous_control/README.md)
- [Algorithm source](../../src/aprenderl/algorithms/ddpg.py)
