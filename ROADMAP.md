# AprendeRL roadmap

Implementation roadmap for reinforcement learning algorithms.
Completed algorithms are checked.

## Classical reinforcement learning

- [x] Multi-armed bandits
- [x] Value iteration
- [x] Policy iteration
- [x] Monte Carlo prediction
- [x] Monte Carlo control
- [x] SARSA - State-Action-Reward-State-Action
- [x] Expected SARSA
- [x] n-step SARSA
- [x] SARSA(λ) and eligibility traces
- [x] Q-learning
- [x] Dyna-Q
- [ ] ARS - Augmented Random Search

## Deep value-based methods

- [x] DQN - Deep Q-Network
- [x] Double DQN
- [x] Dueling DQN
- [x] PER - Prioritized Experience Replay
- [x] n-step DQN
- [x] NoisyNet DQN
- [x] C51 - Categorical DQN with 51 atoms
- [x] Rainbow DQN - Double, Dueling, PER, n-step, NoisyNet, and C51
- [x] QR-DQN - Quantile Regression DQN
- [x] IQN - Implicit Quantile Network

## Policy-gradient methods

- [x] REINFORCE - REward Increment = Nonnegative Factor × Offset Reinforcement
  × Characteristic Eligibility
- [x] REINFORCE with a learned baseline
- [x] Actor-Critic
- [x] A2C - Advantage Actor-Critic
- [ ] A3C - Asynchronous Advantage Actor-Critic
- [x] GAE - Generalized Advantage Estimation
- [ ] NPG - Natural Policy Gradient
- [ ] TRPO - Trust Region Policy Optimization
- [ ] PPO - Proximal Policy Optimization
- [ ] Maskable PPO
- [ ] IMPALA - Importance Weighted Actor-Learner Architecture
- [ ] APPO - Asynchronous Proximal Policy Optimization

## Continuous-control methods

- [ ] DPG - Deterministic Policy Gradient
- [ ] DDPG - Deep Deterministic Policy Gradient
- [ ] TD3 - Twin Delayed DDPG
- [ ] SAC - Soft Actor-Critic
- [ ] SAC with automatic entropy tuning
- [ ] Discrete SAC
- [ ] TQC - Truncated Quantile Critics
- [ ] CrossQ
- [ ] DroQ - Dropout Q-functions

## Goal-conditioned and exploration methods

- [ ] Goal-conditioned DQN
- [ ] Goal-conditioned DDPG
- [ ] HER - Hindsight Experience Replay
- [ ] Goal-conditioned SAC
- [ ] ICM - Intrinsic Curiosity Module
- [ ] RND - Random Network Distillation

## Recurrent methods

- [ ] Recurrent DQN
- [ ] Recurrent A2C
- [ ] Recurrent PPO

## Multi-agent methods

- [ ] Independent DQN
- [ ] Independent PPO
- [ ] Parameter-sharing PPO
- [ ] MADDPG - Multi-Agent DDPG
- [ ] VDN - Value Decomposition Networks
- [ ] QMIX - Q-value mixing network
- [ ] MAPPO - Multi-Agent PPO
- [ ] MASAC - Multi-Agent SAC
- [ ] MATD3 - Multi-Agent TD3

Multi-agent interfaces will be PettingZoo-compatible and support centralized
training with decentralized execution, shared policies, and centralized
critics.

## Imitation and offline reinforcement learning

- [ ] Behavioral cloning
- [ ] DAgger - Dataset Aggregation
- [ ] Advantage-weighted regression
- [ ] TD3+BC - TD3 with Behavioral Cloning
- [ ] CQL - Conservative Q-Learning
- [ ] IQL - Implicit Q-Learning
- [ ] BCQ - Batch-Constrained Q-Learning
- [ ] AWAC - Advantage-Weighted Actor-Critic

## Model-based and advanced methods

- [ ] PETS - Probabilistic Ensembles with Trajectory Sampling
- [ ] MBPO - Model-Based Policy Optimization
- [ ] Dreamer
- [ ] AlphaZero-style MCTS with policy and value learning
- [ ] Decision Transformer
- [ ] Hierarchical reinforcement learning
- [ ] Constrained reinforcement learning
- [ ] Multi-objective reinforcement learning
