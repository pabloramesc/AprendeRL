# AprendeRL

AprendeRL is a lightweight educational reinforcement-learning library written
from scratch with PyTorch and Gymnasium. Its API is intentionally familiar to
Stable-Baselines3 users, while its training loops and update equations remain
small enough to read in one sitting.

Version 0.1 includes Double DQN for a single environment with vector
observations and discrete actions.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Minimal usage

```python
import gymnasium as gym

from aprenderl import DoubleDQN, DoubleDQNConfig
from aprenderl.utils import evaluate_policy

train_env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

agent = DoubleDQN(
    train_env,
    DoubleDQNConfig(seed=42),
    device="auto",
)
agent.learn(total_timesteps=20_000)
result = evaluate_policy(agent, eval_env, episodes=10, seed=1_042)
agent.save("artifacts/cartpole.pt")

print(result.mean_return)
train_env.close()
eval_env.close()
```

Double DQN uses the online network to select the next action and the target
network to evaluate it:

```text
y = reward + gamma * (1 - terminated)
    * Q_target(next_state, argmax_a Q_online(next_state, a))
```

This keeps action selection separate from target evaluation and reduces the
overestimation produced by vanilla DQN's maximization target.

## CartPole example

```bash
python examples/train_cartpole.py --timesteps 20000
```

The script reports training and evaluation returns and writes a checkpoint to
`artifacts/double_dqn_cartpole.pt` by default.

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

Gymnasium's `terminated` and `truncated` signals are stored separately. Double
DQN stops bootstrapping only at true terminal states; a time-limit truncation
still receives a value target. Evaluation uses a separate environment so it
cannot disturb training state.

See [ROADMAP.md](ROADMAP.md) for the planned classical RL, DQN, policy-gradient,
continuous-control, offline, model-based, recurrent, and multi-agent phases.
