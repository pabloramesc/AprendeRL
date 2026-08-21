# REINFORCE

[Back to the algorithm index](../algorithms.md)

**Policy classification:** On-policy. Every update uses complete trajectories
sampled from the current stochastic policy, before that policy changes.

**Use when:** observations are numeric `Box` vectors or tensors and actions are
either discrete or a finite or fully unbounded continuous `Box`, and a compact,
educational on-policy algorithm is more important than sample efficiency. It
can run either as vanilla REINFORCE or with a learned state-value baseline.

## Policy and returns

REINFORCE parameterizes a stochastic policy $\pi_\theta(a\mid s)$. For
`Discrete` actions, the default network returns categorical logits. For finite
`Box` actions, `GaussianPolicyNetwork` returns means and learned,
state-independent log standard deviations. Finite action ranges pass samples
through `tanh` and rescale them to the environment bounds, so executed actions
and their log probabilities remain consistent without clipping. Fully
unbounded spaces use the Gaussian samples directly.

For each completed episode, AprendeRL computes discounted reward-to-go:

$$
G_t = \sum_{k=t}^{T-1}\gamma^{k-t}r_{k+1}.
$$

## Policy-gradient loss

The algorithm minimizes the negative Monte Carlo policy-gradient objective over
all $N$ transitions collected from `episodes_per_update` episodes:

$$
\mathcal{L}_{\mathrm{policy}}(\theta)
= -\frac{1}{N}\sum_{t=0}^{N-1}
G_t\log\pi_\theta(a_t\mid s_t).
$$

Vanilla REINFORCE is the default. Set `use_baseline=True` to learn a separate
state-value network by Monte Carlo regression:

$$
\mathcal{L}_{\mathrm{value}}(\phi)
= \frac{1}{N}\sum_{t=0}^{N-1}\left(V_\phi(s_t)-G_t\right)^2.
$$

The policy is then weighted by the detached advantage estimate
$A_t=G_t-V_\phi(s_t)$ instead of $G_t$. The value target remains the raw return,
so normalizing the policy weights cannot change the scale learned by the value
network.

By default, policy weights are standardized within each update batch: returns
in vanilla mode and advantages in baseline mode. This does not change their
ordering and usually reduces gradient variance. An optional entropy bonus
encourages broader action distributions:

$$
\mathcal{L}(\theta)=\mathcal{L}_{\mathrm{policy}}(\theta)
-\beta\frac{1}{N}\sum_t\mathcal{H}\left(\pi_\theta(\cdot\mid s_t)\right).
$$

Adam updates the policy and, when enabled, the value network using independent
optimizers. Their learning rates are `learning_rate` and `value_learning_rate`;
both gradient norms are clipped to `max_grad_norm`. Both variants wait for full
episodes and do not bootstrap at time-limit truncations. For continuous
policies, the entropy metric and bonus use the simple diagonal-Gaussian entropy;
for finite actions this is the pre-squash entropy.

## Switching on the baseline

```python
config = REINFORCEConfig(
    use_baseline=True,
    learning_rate=1e-2,
    value_learning_rate=1e-2,
)
agent = REINFORCE(env, config=config)
```

For continuous actions, the same class selects the Gaussian policy from the
environment:

```python
env = gym.make("Pendulum-v1")
config = REINFORCEConfig(
    learning_rate=3e-4,
    value_learning_rate=1e-3,
    episodes_per_update=4,
    use_baseline=True,
)
agent = REINFORCE(env, config=config).learn(50_000)
```

With `use_baseline=False` (the default), no value network is created and saved
checkpoints retain the vanilla behavior. A custom baseline can be supplied as
`value_network=...`; it must map an observation batch to a tensor of shape
`(batch_size,)`. A custom discrete policy network returns logits with shape
`(batch_size, action_count)`. A custom continuous policy network returns a
`(means, log_stds)` tuple; both tensors have shape
`(batch_size, flattened_action_size)`.

The implementation is in
[`src/aprenderl/algorithms/reinforce.py`](../../src/aprenderl/algorithms/reinforce.py),
and the runnable example is
[`examples/train_reinforce.ipynb`](../../examples/train_reinforce.ipynb). The
continuous example is
[`examples/train_reinforce_pendulum.ipynb`](../../examples/train_reinforce_pendulum.ipynb).
