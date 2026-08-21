# AprendeRL

AprendeRL is a lightweight educational reinforcement-learning library written
from scratch with PyTorch and Gymnasium. Its API is intentionally familiar to
Stable-Baselines3 users, while its training loops and update equations remain
small enough to read in one sitting.

The implemented algorithms are off-policy tabular Q-Learning and vanilla DQN,
plus the on-policy tabular SARSA, REINFORCE, and one-step Actor-Critic methods.
REINFORCE supports both its vanilla form and an optional learned state-value
baseline.

## Install

```bash
python -m pip install -e .
```

## Minimal usage

Train and evaluate a DQN agent in a few lines:

```python
import gymnasium as gym
from aprenderl import DQN
from aprenderl.utils import evaluate_policy

env = gym.make("CartPole-v1")

agent = DQN(env).learn(20_000)
result = evaluate_policy(agent, env, episodes=10)

print(result.mean_return)
env.close()
```

Pass any compatible PyTorch module to customize the Q-network. It must map a
batch of observations to one value per discrete action:

```python
from torch import nn

network = nn.Sequential(
    nn.Flatten(),
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)
agent = DQN(train_env, network)
```

When `network` is omitted, AprendeRL creates its default MLP. A target network
is made by deep-copying the supplied module.

## Example notebooks

- [DQN](examples/train_dqn.ipynb)
- [Tabular Q-Learning](examples/train_qlearning.ipynb)
- [Tabular SARSA](examples/train_sarsa.ipynb)
- [REINFORCE](examples/train_reinforce.ipynb)
- [Actor-Critic](examples/train_actor_critic.ipynb)

Each notebook defines its Gymnasium environment with an `ENV_ID` constant near
the beginning, so you can switch to another environment compatible with the
algorithm's observation and action spaces.

## Repository layout

```text
src/aprenderl/
├── algorithms/      # Interfaces, value-based methods, and policy gradients
├── buffers/         # Replay and ordered rollout buffers
├── callbacks/       # Training lifecycle hooks
├── distributions/   # Reusable action distributions
├── envs/            # Gymnasium wrappers
├── logging/         # Scalar metric collection
├── networks/        # PyTorch policy, value, and Q-network components
├── policies/        # Exploration components and schedules
├── config.py        # Experiment configuration
├── types.py         # Transitions and episode metrics
└── utils/           # Seeding, devices, and evaluation
```

Gymnasium's `terminated` and `truncated` signals are stored separately.
Value-based algorithms and Actor-Critic stop bootstrapping only at true terminal
states; REINFORCE updates only after a complete Gymnasium episode. Use a
separate evaluation environment if training will continue afterward, so
evaluation does not disturb the training state.

See the [algorithm documentation index](docs/algorithms.md) for notation,
off-policy and on-policy classifications, and a separate mathematical guide for
every implemented algorithm.

See [ROADMAP.md](ROADMAP.md) for the planned classical RL, DQN, policy-gradient,
continuous-control, offline, model-based, recurrent, and multi-agent phases.
