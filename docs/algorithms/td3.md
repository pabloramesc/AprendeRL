# Twin Delayed DDPG (TD3)

TD3 extends [DDPG](ddpg.md) with two independent critics, target-policy
smoothing, and delayed actor updates. These mechanisms address critic
approximation error and overestimation.

## Learning rule

For replay transitions $(s,a,r,s',d)$, $d$ denotes true termination and
$\gamma$ the discount. Actor parameters are $\theta$, critic parameters are
$\phi_1,\phi_2$, and bars indicate target copies. Work in normalized action
coordinates for smoothing:

$$
\epsilon\sim\operatorname{clip}(\mathcal N(0,\sigma^2I),-c,c),\qquad
\tilde u=\operatorname{clip}(\mu_{\bar\theta}^{\mathrm{normalized}}(s')
+\epsilon,-1,1).
$$

Rescale $\tilde u$ to environment action bounds to obtain $\tilde a$.
Here $\sigma$ is `target_policy_noise`, $c$ is `target_noise_clip`, and $I$
is the identity covariance. The Bellman target uses the smaller target Q:

$$
y=r+\gamma(1-d)\min_{i\in\{1,2\}}Q_{\bar\phi_i}(s',\tilde a),\qquad
L_Q=\sum_{i=1}^2\mathbb E[(Q_{\phi_i}(s,a)-y)^2].
$$

Every `policy_delay` critic updates, improve the actor using the **first**
online critic, and update both target critics and the target actor:

$$
L_\mu=-\mathbb E[Q_{\phi_1}(s,\mu_\theta(s))],\qquad
\bar w\leftarrow(1-\tau)\bar w+\tau w.
$$

$w$ denotes online parameters. The delay counts gradient updates, including
multiple `gradient_steps` within one environment step and across successive
`learn` calls or restored checkpoints. Target networks remain fixed between
actor updates. Critic parameters are frozen during actor optimization, while
the action derivative remains available. Targets are computed without gradients.

## Exploration and bounds

Behavior uses the same uniform warmup and normalized Gaussian action noise
as DDPG. Target smoothing is separate noise used only in Bellman targets.
Both noise mechanisms respect asymmetric, per-dimension action bounds.
`predict(..., deterministic=True)` disables behavior noise.

## Public API

```python
import gymnasium as gym
from aprenderl import TD3, TD3Config
from aprenderl.utils import evaluate_policy

config = TD3Config(batch_size=128, learning_starts=1_000, seed=7)
env = gym.make("Pendulum-v1")
try:
    agent = TD3(env, config=config, device="cpu")
    agent.learn(total_timesteps=20_000)
    agent.save("artifacts/td3.pt")
finally:
    env.close()

evaluation_env = gym.make("Pendulum-v1")
try:
    restored = TD3.load("artifacts/td3.pt", evaluation_env, device="cpu")
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

- [Fujimoto et al.](https://proceedings.mlr.press/v80/fujimoto18a.html)
- [Public API example](../../examples/td3.ipynb)
- [From-scratch study notebook](../../study/07_continuous_control/03_td3.ipynb)
- [Continuous-control chapter](../../study/07_continuous_control/README.md)
- [Algorithm source](../../src/aprenderl/algorithms/td3.py)
