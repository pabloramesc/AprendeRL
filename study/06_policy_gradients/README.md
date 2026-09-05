# Chapter 6: Policy gradients

## Learning objectives

- Optimize a stochastic policy directly.
- Use a learned value baseline to reduce variance.
- Handle discrete and continuous action spaces.
- Extend one-step actor-critic to multi-step targets.
- Constrain policy updates with TRPO, conjugate gradients, and a KL line search.
- Learn PPO clipping and repeated minibatch optimization of on-policy rollouts.

## Suggested prerequisite

- [Temporal-difference learning](../03_temporal_difference/README.md)

## Notebooks

1. [REINFORCE](01_reinforce.ipynb)
2. [REINFORCE with continuous actions](02_reinforce_continuous.ipynb)
3. [Actor-Critic](03_actor_critic.ipynb)
4. [Actor-Critic with continuous actions](04_actor_critic_continuous.ipynb)
5. [n-step Actor-Critic](05_actor_critic_n_step.ipynb)
6. [Trust Region Policy Optimization (TRPO)](06_trpo.ipynb)
7. [Proximal Policy Optimization (PPO)](07_ppo.ipynb)

Study n-step Actor-Critic before TRPO: TRPO builds on rollouts and value
baselines, introducing generalized advantage estimation and constrained policy
optimization. Then study PPO to reuse those estimates with a clipped objective
and shuffled minibatch updates.
