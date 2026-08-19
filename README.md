# AprendeRL

AprendeRL is a lightweight educational reinforcement-learning library written
from scratch with PyTorch and Gymnasium. Its API is intentionally familiar to
Stable-Baselines3 users, while its training loops and update equations remain
small enough to read in one sitting.

Version 0.1 includes vanilla DQN for a single environment with vector
observations and discrete actions.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Minimal usage

```python
import gymnasium as gym

from aprenderl import DQN, DQNConfig
from aprenderl.utils import evaluate_policy

train_env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

agent = DQN(
    train_env,
    config=DQNConfig(seed=42),
    device="auto",
)
agent.learn(total_timesteps=20_000)
result = evaluate_policy(agent, eval_env, episodes=10, seed=1_042)
agent.save("artifacts/cartpole.pt")

print(result.mean_return)
train_env.close()
eval_env.close()
```

DQN uses a separate target network to construct stable temporal-difference
targets:

```text
y = reward + gamma * (1 - terminated) * max_a Q_target(next_state, a)
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

## CartPole example

```bash
python examples/train_cartpole.py --timesteps 20000
```

The script reports training and evaluation returns and writes a checkpoint to
`artifacts/dqn_cartpole.pt` by default.

## Repository layout

```text
src/aprenderl/
├── algorithms/      # Interfaces, reusable loops, and DQN updates
├── buffers/         # Replay and ordered rollout buffers
├── callbacks/       # Training lifecycle hooks
├── distributions/   # Reusable action distributions
├── envs/            # Gymnasium wrappers
├── logging/         # Scalar metric collection
├── networks/        # PyTorch Q-network components
├── policies/        # Exploration components and schedules
├── config.py        # Experiment configuration
├── types.py         # Transitions and episode metrics
└── utils/           # Seeding, devices, and evaluation
```

Gymnasium's `terminated` and `truncated` signals are stored separately. DQN
stops bootstrapping only at true terminal states; a time-limit truncation still
receives a value target. Evaluation uses a separate environment so it cannot
disturb training state.

See [ROADMAP.md](ROADMAP.md) for the planned classical RL, DQN, policy-gradient,
continuous-control, offline, model-based, recurrent, and multi-agent phases.
