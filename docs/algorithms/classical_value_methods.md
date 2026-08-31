# Classical value-based methods

[Back to the algorithm index](../algorithms.md)

This guide covers the finite-state methods that complement Q-Learning and
one-step SARSA. All interaction-based agents require discrete Gymnasium action
spaces; tabular control and prediction also require discrete observations.
Planning additionally requires `env.unwrapped.P`, the transition model exposed
by environments such as `FrozenLake-v1`.

## Multi-armed bandits

`MultiArmedBandit` ignores the observation and maintains one estimate per arm.
With the default sample-average step size, the selected arm is updated by

$$
Q_{n+1}(a)=Q_n(a)+\frac{1}{N_n(a)}[R_n-Q_n(a)].
$$

Set `learning_rate` to a constant for a recency-weighted estimate in a
non-stationary bandit. Exploration follows the same linear epsilon schedule as
the tabular control algorithms.

## Value iteration

`ValueIteration` repeatedly applies the Bellman optimality operator:

$$
V_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma(1-d)V_k(s')\right].
$$

Sweeps stop when the largest value change is below `tolerance`, or after the
number of sweeps passed to `learn`. A final greedy improvement fills `policy`.

## Policy iteration

`PolicyIteration` alternates two explicit phases:

1. evaluate the current deterministic policy until its Bellman residual is
   below `tolerance`;
2. replace every action with a greedy action under the evaluated values.

It stops once improvement leaves the complete policy unchanged. For both
planning methods, `num_timesteps` counts planning sweeps or improvements—not
environment interactions.

## Monte Carlo prediction

`MonteCarloPrediction(policy, env)` estimates $V^\pi$ from complete episodes.
The fixed policy may be a callable, a vector of deterministic actions, or a
matrix of action probabilities. The return is

$$
G_t=R_{t+1}+\gamma R_{t+2}+\cdots,
$$

and first-visit mode updates only the earliest occurrence of a state in each
episode. Every-visit mode updates every occurrence. A time-limit truncation
bootstraps from the current estimate at its final observation.

## Monte Carlo control

`MonteCarloControl` learns $Q$ from complete epsilon-greedy episodes. It uses
the incremental average of the sampled returns for each state-action pair by
default; set `learning_rate` for a constant step size. It obtains its
deterministic policy by taking `argmax` over `q_table`.

## Expected SARSA

Expected SARSA replaces SARSA's sampled next action with its exact expectation
under the current epsilon-greedy policy:

$$
y_t=R_{t+1}+\gamma(1-d_t)
\sum_a\pi_\varepsilon(a\mid S_{t+1})Q(S_{t+1},a).
$$

For one greedy action and $|\mathcal A|$ total actions, its probability is
$1-\varepsilon+\varepsilon/|\mathcal A|$; every other action has probability
$\varepsilon/|\mathcal A|$.

## n-step SARSA

`NStepSARSA` delays an update until it can form

$$
G_{t:t+n}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+\gamma^nQ(S_{t+n},A_{t+n}).
$$

At an episode boundary it flushes all shorter remaining returns. True terminal
states omit the bootstrap; time-limit truncations retain it.

## SARSA(lambda)

`SARSALambda` uses accumulating eligibility traces:

$$
E_t(s,a)=\gamma\lambda E_{t-1}(s,a)
+\mathbf 1\{s=S_t,a=A_t\},
$$

$$
Q \leftarrow Q+\alpha\delta_tE_t.
$$

Traces reset at every Gymnasium episode boundary. `trace_decay=0` recovers the
one-step SARSA update.

## Dyna-Q

`DynaQ` performs one ordinary Q-Learning update, stores the observed one-step
transition in a deterministic tabular model, and then performs
`planning_steps` Q-Learning updates sampled from that model. Thus one real
transition produces `1 + planning_steps` value updates once the model is
non-empty.

## Minimal API

```python
import gymnasium as gym

from aprenderl import DynaQ, DynaQConfig

env = gym.make("FrozenLake-v1", is_slippery=False)
agent = DynaQ(env, config=DynaQConfig(planning_steps=10))
agent.learn(5_000)

action = agent.predict(0, deterministic=True)
agent.save("dyna_q.npz")
restored = DynaQ.load("dyna_q.npz", env)
```

The implementations live in
[`src/aprenderl/algorithms`](../../src/aprenderl/algorithms).
