# AprendeRL roadmap

The 0.1 release establishes the small, stable core used by later algorithms:
Gymnasium interaction, typed configuration, PyTorch networks, replay storage,
callbacks, evaluation, logging, checkpointing, and vanilla DQN. Later
components will be added when an implemented algorithm needs them, keeping the
mathematics visible and avoiding framework machinery for its own sake.

## Planned phases

### 1. Classical RL

Bandits, value iteration, policy iteration, Monte Carlo control, SARSA, and
Q-learning.

### 2. DQN family

Double DQN, Dueling DQN, prioritized replay, n-step returns, NoisyNet, C51,
QR-DQN, and Rainbow.

### 3. Policy gradients

REINFORCE, learned baselines, actor–critic, A2C, GAE, and PPO.

### 4. Continuous control

DDPG, TD3, SAC, automatic entropy tuning, and discrete SAC. This phase adds
Gaussian, squashed-Gaussian, and deterministic action components.

### 5. Goal-conditioned and exploration methods

HER, goal-conditioned DQN/SAC, intrinsic curiosity modules (ICM), and random
network distillation (RND).

### 6. Partial observability

Recurrent DQN, A2C, and PPO using GRU/LSTM policies and sequence-aware buffers.

### 7. Multi-agent reinforcement learning

Independent DQN/PPO, parameter-sharing PPO, MAPPO, MADDPG, VDN, QMIX, MASAC,
and MATD3. Interfaces will be PettingZoo-compatible and support centralized
training with decentralized execution (CTDE), shared policies, and centralized
critics.

### 8. Imitation and offline RL

Behavioral cloning, DAgger, TD3+BC, CQL, IQL, and advantage-weighted regression.

### 9. Advanced methods

Dyna-Q, PETS, MBPO, Dreamer, Decision Transformer, hierarchical RL, constrained
RL, and multi-objective RL.

## Recommended release sequence

| Release | Scope |
| --- | --- |
| `0.1` | Core API and vanilla DQN |
| `0.2` | Double/Dueling DQN, prioritized replay, and n-step returns |
| `0.3` | REINFORCE, A2C, and GAE |
| `0.4` | PPO and vectorized environments |
| `0.5` | DDPG, TD3, and SAC |
| `0.6` | C51, NoisyNet, and Rainbow |
| `0.7` | Goal-conditioned RL and HER |
| `0.8` | Recurrent policies |
| `0.9` | IPPO, MAPPO, VDN, and QMIX |
| `1.0` | Stable API, documentation, and reproducible benchmarks |
