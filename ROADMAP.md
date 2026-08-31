# AprendeRL roadmap

Implementation roadmap for reinforcement learning algorithms.
Completed algorithms are checked.

## Classical reinforcement learning

- [ ] Multi-armed bandits
- [ ] Value iteration
- [ ] Policy iteration
- [ ] Monte Carlo prediction
- [ ] Monte Carlo control
- [x] SARSA — State-Action-Reward-State-Action
- [ ] n-step SARSA
- [x] Q-learning
- [ ] Dyna-Q

## Deep value-based methods

- [x] DQN — Deep Q-Network
- [ ] Double DQN
- [ ] Dueling DQN
- [ ] Prioritized experience replay
- [ ] n-step DQN
- [ ] NoisyNet DQN
- [ ] C51 — Categorical DQN with 51 atoms
- [ ] QR-DQN — Quantile Regression DQN
- [ ] Rainbow

## Policy-gradient methods

- [x] REINFORCE — REward Increment = Nonnegative Factor × Offset Reinforcement
  × Characteristic Eligibility
- [x] REINFORCE with a learned baseline
- [x] Actor-Critic
- [x] A2C — Advantage Actor-Critic
- [x] GAE — Generalized Advantage Estimation
- [ ] TRPO — Trust Region Policy Optimization
- [ ] PPO — Proximal Policy Optimization

## Continuous-control methods

- [ ] DPG — Deterministic Policy Gradient
- [ ] DDPG — Deep Deterministic Policy Gradient
- [ ] TD3 — Twin Delayed DDPG
- [ ] SAC — Soft Actor-Critic
- [ ] SAC with automatic entropy tuning
- [ ] Discrete SAC

## Goal-conditioned and exploration methods

- [ ] Goal-conditioned DQN
- [ ] Goal-conditioned DDPG
- [ ] HER — Hindsight Experience Replay
- [ ] Goal-conditioned SAC
- [ ] ICM — Intrinsic Curiosity Module
- [ ] RND — Random Network Distillation

## Recurrent methods

- [ ] Recurrent DQN
- [ ] Recurrent A2C
- [ ] Recurrent PPO

## Multi-agent methods

- [ ] Independent DQN
- [ ] Independent PPO
- [ ] Parameter-sharing PPO
- [ ] MADDPG — Multi-Agent DDPG
- [ ] VDN — Value Decomposition Networks
- [ ] QMIX — Q-value mixing network
- [ ] MAPPO — Multi-Agent PPO
- [ ] MASAC — Multi-Agent SAC
- [ ] MATD3 — Multi-Agent TD3

Multi-agent interfaces will be PettingZoo-compatible and support centralized
training with decentralized execution, shared policies, and centralized
critics.

## Imitation and offline reinforcement learning

- [ ] Behavioral cloning
- [ ] DAgger — Dataset Aggregation
- [ ] Advantage-weighted regression
- [ ] TD3+BC — TD3 with Behavioral Cloning
- [ ] CQL — Conservative Q-Learning
- [ ] IQL — Implicit Q-Learning

## Model-based and advanced methods

- [ ] PETS — Probabilistic Ensembles with Trajectory Sampling
- [ ] MBPO — Model-Based Policy Optimization
- [ ] Dreamer
- [ ] Decision Transformer
- [ ] Hierarchical reinforcement learning
- [ ] Constrained reinforcement learning
- [ ] Multi-objective reinforcement learning
