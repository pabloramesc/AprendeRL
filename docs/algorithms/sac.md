# Soft Actor-Critic (SAC)

SAC learns a stochastic actor from replay while rewarding both return and
policy entropy. This implementation follows the updated formulation with two
Q critics, target critics, and optional automatic temperature tuning. It has
no separate value network or target actor. The original
[2018 SAC formulation](https://proceedings.mlr.press/v80/haarnoja18b.html)
included an explicit value-function approximation.

## Objective and policy

For observation $s_t$, action $a_t$, reward $r_{t+1}$, discount $\gamma$, and
stochastic policy $\pi_\theta$, the maximum-entropy objective is

$$J(\pi)=\mathbb E\left[\sum_{t\ge0}\gamma^t
(r_{t+1}+\alpha\mathcal H(\pi(\cdot\mid s_t)))\right].$$

$\mathcal H(\pi)=-\mathbb E_a\log\pi(a\mid s)$ is differential entropy and
$\alpha$ its temperature. The actor outputs state-dependent means $m$ and
log standard deviations, clamped to $[-20,2]$. For each action dimension:

$$u=m_\theta(s)+\sigma_\theta(s)\epsilon,\quad
\epsilon\sim\mathcal N(0,I),\quad a=b+c\tanh u.$$

For bounds $l,h$, $b=(h+l)/2$ and $c=(h-l)/2$. Reparameterized sampling retains
gradients from the critic through actions into actor parameters. The density
includes both transformations:

$$\log\pi(a\mid s)=\sum_j\left[
\log\mathcal N(u_j;m_j,\sigma_j)-\log c_j-\log(1-\tanh^2u_j)\right].$$

The implementation computes the tanh Jacobian from the raw sample with a
stable softplus identity, avoiding inversion of saturated actions. Index
$j$ runs over flattened action dimensions, and $I$ is the identity covariance.

## Learning rule

Let $(s,a,r,s',d)$ be a replay transition, with $d$ indicating true termination.
The two critic parameter sets are $\phi_i$; bars indicate target copies.
Draw $a'\sim\pi_\theta(\cdot\mid s')$ and form the detached target:

$$y=r+\gamma(1-d)\left[\min_iQ_{\bar\phi_i}(s',a')
-\alpha\log\pi_\theta(a'\mid s')\right],\qquad
L_Q=\sum_i\mathbb E[(Q_{\phi_i}(s,a)-y)^2].$$

After the critic step, sample current policy actions at replay observations:

$$L_\pi=\mathbb E_{s,a\sim\pi_\theta}
[\alpha\log\pi_\theta(a\mid s)-\min_iQ_{\phi_i}(s,a)].$$

Here states come from replay and actions from the current policy. Freeze
critic weights, retaining action gradients. Target critics move after every
update: $\bar\phi_i\leftarrow(1-\tau)\bar\phi_i+\tau\phi_i$.

## Temperature and entropy units

`ent_coef="auto"` learns $\beta=\log\alpha$ using the usual log-temperature
surrogate:

$$L_\beta=-\mathbb E[\beta\operatorname{stopgrad}
(\log\pi_\theta(a\mid s)+\mathcal H_*)].$$

$\mathcal H_*$ is `target_entropy`. This increases temperature when estimated
entropy is too low and decreases it when entropy is too high. Actor/critic
updates use detached $\alpha$. `initial_ent_coef` starts automatic tuning;
`ent_coef_learning_rate` controls its Adam optimizer. A numeric `ent_coef`,
including zero, fixes the temperature and disables that optimizer.

Densities and explicit entropy targets are in **environment action units**.
The default target is $-n+\sum_j\log c_j$, where $n$ is the flattened action
dimension. This is the familiar normalized target $-n$, shifted by the affine
Jacobian. Differential entropy can be negative. Temperature tuning therefore
uses consistent units even for asymmetric or differently scaled action bounds.
The soft value itself includes the corresponding entropy offset.

Behavior uses uniform warmup, then samples the learned policy; no extra action
noise is added. Deterministic prediction squashes and rescales the Gaussian
mean. It is an evaluation convention, not the mean of the transformed density.

## Public API

```python
import gymnasium as gym
from aprenderl import SAC, SACConfig
from aprenderl.utils import evaluate_policy

config = SACConfig(batch_size=128, learning_starts=1_000, seed=7)
env = gym.make("Pendulum-v1")
try:
    agent = SAC(env, config=config, device="cpu")
    agent.learn(total_timesteps=20_000)
    agent.save("artifacts/sac.pt")
finally:
    env.close()

evaluation_env = gym.make("Pendulum-v1")
try:
    restored = SAC.load("artifacts/sac.pt", evaluation_env, device="cpu")
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
Supply `policy_network` returning `(means, log_stds)`, each of shape
`(batch, action_dim)`. The default `SACPolicyNetwork` has state-dependent
standard deviations, unlike the on-policy default Gaussian network.
Supply `critic_networks` as a sequence of 2 independent modules.
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

- [Haarnoja et al., Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- [Public API example](../../examples/sac.ipynb)
- [From-scratch study notebook](../../study/07_continuous_control/04_sac.ipynb)
- [Continuous-control chapter](../../study/07_continuous_control/README.md)
- [Algorithm source](../../src/aprenderl/algorithms/sac.py)
