# AprendeRL roadmap

Implementation roadmap for reinforcement learning algorithms.
Completed algorithms are checked.

## Classical reinforcement learning

- [x] Q-learning (Watkins and Dayan, 1992)
- [x] SARSA - State-Action-Reward-State-Action (Rummery and Niranjan, 1994)

## Deep value-based methods

- [x] DQN - Deep Q-Network, with Double DQN targets
  (Mnih et al., 2015; van Hasselt et al., 2016)
- [x] C51 - Categorical DQN with 51 atoms (Bellemare et al., 2017)
- [x] Rainbow DQN - Double, Dueling, PER, n-step, NoisyNet, and C51
  (Hessel et al., 2018)
- [x] QR-DQN - Quantile Regression DQN (Dabney et al., 2018)
- [x] IQN - Implicit Quantile Network (Dabney et al., 2018)
- [x] FQF - Fully Parameterized Quantile Function (Yang et al., 2019)

## Policy-gradient methods

- [x] REINFORCE - Monte Carlo policy gradient, with optional baselines
  (Williams, 1992)
- [x] Actor-Critic (Barto et al., 1983; Sutton et al., 2000)
- [x] A2C - Advantage Actor-Critic
  (Mnih et al., 2016; Schulman et al., 2016)
- [ ] TRPO - Trust Region Policy Optimization (Schulman et al., 2015)
- [ ] PPO - Proximal Policy Optimization (Schulman et al., 2017)

## Continuous-control methods

- [ ] DPG - Deterministic Policy Gradient (Silver et al., 2014)
- [ ] DDPG - Deep Deterministic Policy Gradient (Lillicrap et al., 2016)
- [ ] TD3 - Twin Delayed DDPG (Fujimoto et al., 2018)
- [ ] SAC - Soft Actor-Critic (Haarnoja et al., 2018)
- [ ] Discrete SAC (Christodoulou, 2019)

## Goal-conditioned and exploration methods

- [ ] Goal-conditioned DQN (Schaul et al., 2015)
- [ ] Goal-conditioned DDPG (Andrychowicz et al., 2017)
- [ ] HER - Hindsight Experience Replay (Andrychowicz et al., 2017)
- [ ] Goal-conditioned SAC (Schaul et al., 2015; Haarnoja et al., 2018)
- [ ] ICM - Intrinsic Curiosity Module (Pathak et al., 2017)
- [ ] RND - Random Network Distillation (Burda et al., 2019)

## Recurrent methods

- [ ] Recurrent DQN (Hausknecht and Stone, 2015)
- [ ] Recurrent A2C (Mnih et al., 2016)
- [ ] Recurrent PPO (Schulman et al., 2017)

## Distributed architectures

- [ ] A3C - Asynchronous Advantage Actor-Critic (Mnih et al., 2016)
- [ ] IMPALA - Importance Weighted Actor-Learner Architecture
  (Espeholt et al., 2018)
- [ ] APPO - Asynchronous Proximal Policy Optimization (Petrenko et al., 2020)

## Multi-agent methods

- [ ] Independent DQN (Tan, 1993; Mnih et al., 2015)
- [ ] Independent PPO (de Witt et al., 2020)
- [ ] Parameter-sharing PPO (Schulman et al., 2017)
- [ ] MADDPG - Multi-Agent DDPG (Lowe et al., 2017)
- [ ] VDN - Value Decomposition Networks (Sunehag et al., 2018)
- [ ] QMIX - Q-value mixing network (Rashid et al., 2018)
- [ ] MAPPO - Multi-Agent PPO (Yu et al., 2022)
- [ ] MASAC - Multi-Agent SAC (Haarnoja et al., 2018; Lowe et al., 2017)
- [ ] MATD3 - Multi-Agent TD3 (Ackermann et al., 2019)

Multi-agent interfaces will be PettingZoo-compatible and support centralized
training with decentralized execution, shared policies, and centralized
critics.

## Imitation and offline reinforcement learning

- [ ] Behavioral cloning (Bain and Sammut, 1995)
- [ ] DAgger - Dataset Aggregation (Ross et al., 2011)
- [ ] Advantage-weighted regression (Peng et al., 2019)
- [ ] TD3+BC - TD3 with Behavioral Cloning (Fujimoto and Gu, 2021)
- [ ] CQL - Conservative Q-Learning (Kumar et al., 2020)
- [ ] IQL - Implicit Q-Learning (Kostrikov et al., 2022)
- [ ] BCQ - Batch-Constrained Q-Learning (Fujimoto et al., 2019)
- [ ] AWAC - Advantage-Weighted Actor-Critic (Nair et al., 2020)

## Model-based and advanced methods

- [ ] PETS - Probabilistic Ensembles with Trajectory Sampling (Chua et al., 2018)
- [ ] MBPO - Model-Based Policy Optimization (Janner et al., 2019)
- [ ] Dreamer (Hafner et al., 2020)
- [ ] AlphaZero-style MCTS with policy and value learning (Silver et al., 2018)
- [ ] Decision Transformer (Chen et al., 2021)
- [ ] Hierarchical reinforcement learning (Sutton et al., 1999)
- [ ] Constrained reinforcement learning (Achiam et al., 2017)
- [ ] Multi-objective reinforcement learning (Roijers et al., 2013)
