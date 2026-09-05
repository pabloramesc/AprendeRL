# AprendeRL

AprendeRL is a lightweight educational reinforcement-learning library written
from scratch with PyTorch and Gymnasium. Its API is intentionally familiar to
Stable-Baselines3 users, while its training loops and update equations remain
small enough to read in one sitting.

## Quick install

### Using pip

Create a standard Python virtual environment, activate it, and install the
project with `pip`:

```bash
git clone https://github.com/pabloramesc/AprendeRL.git
cd AprendeRL
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

### Using uv

Alternatively, with [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone https://github.com/pabloramesc/AprendeRL.git
cd AprendeRL
uv sync
```

`uv sync` creates the `.venv` virtual environment and installs AprendeRL with
its locked dependencies. It may not install a `pip` executable, so use one
complete workflow or the other.

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
agent = DQN(env, network)
```

When `network` is omitted, AprendeRL creates its default MLP. A target network
is made by deep-copying the supplied module.

## Algorithm coverage

The public library follows a compact learning path:

- tabular Q-Learning and SARSA;
- DQN with optional Double DQN targets;
- the paper-level distributional agents C51, Rainbow DQN, QR-DQN, IQN, and FQF;
- REINFORCE, Actor-Critic, A2C, TRPO, and PPO for discrete and continuous actions.

Every algorithm follows `learn`, `predict`, `save`, and `load`. Smaller
foundational methods and individual DQN mechanisms remain as self-contained
study notebooks rather than additional public classes.

Shared environment interaction and training state live in `BaseAlgorithm`.
`OnPolicyAlgorithm` and `OffPolicyAlgorithm` identify the learning family,
while `PolicyGradientAlgorithm` contains the neural policy and action-space
machinery shared by REINFORCE, Actor-Critic, A2C, TRPO, and PPO. The mathematical
update for each algorithm remains in its own module.

## Example notebooks

Complete training examples:

- [Tabular Q-Learning](examples/qlearning.ipynb)
- [Tabular SARSA](examples/sarsa.ipynb)
- [DQN with Double DQN targets](examples/dqn.ipynb)
- [C51](examples/c51.ipynb)
- [Rainbow DQN](examples/rainbow_dqn.ipynb)
- [QR-DQN](examples/qr_dqn.ipynb)
- [IQN](examples/iqn.ipynb)
- [FQF](examples/fqf.ipynb)
- [REINFORCE](examples/reinforce.ipynb)
- [REINFORCE on continuous actions](examples/reinforce_continuous.ipynb)
- [Actor-Critic](examples/actor_critic.ipynb)
- [Actor-Critic on continuous actions](examples/actor_critic_continuous.ipynb)
- [A2C](examples/a2c.ipynb)
- [A2C on continuous actions](examples/a2c_continuous.ipynb)
- [TRPO](examples/trpo.ipynb)
- [PPO](examples/ppo.ipynb)

Each notebook defines its Gymnasium environment with an `ENV_ID` constant near
the beginning, so you can switch to another environment compatible with the
algorithm's observation and action spaces.

Regenerate and keep the outputs of every example notebook with:

```bash
./scripts/regenerate_example_outputs.sh
```

Set `NOTEBOOK_CELL_TIMEOUT` to change the default 900-second timeout per cell.

## Study notebooks

These notebooks implement concepts directly with Gymnasium, NumPy, and
PyTorch. They are organized as a seven-chapter learning path, from bandits and
dynamic programming through value-based deep RL and policy gradients. Start
with the [study guide](study/README.md).

Regenerate and keep the outputs of every study notebook with:

```bash
./scripts/regenerate_study_outputs.sh
```

This script also honors the `NOTEBOOK_CELL_TIMEOUT` setting described above.

## Repository layout

```text
src/aprenderl/
├── algorithms/      # Core algorithms and the paper-level DQN family
├── buffers/         # Replay and ordered rollout buffers
├── callbacks/       # Training lifecycle hooks
├── distributions/   # Categorical, Gaussian, and squashed Gaussian policies
├── envs/            # Gymnasium wrappers
├── logging/         # Scalar metric collection
├── networks/        # PyTorch policy, value, and Q-network components
├── policies/        # Exploration components and schedules
├── config.py        # Experiment configuration
├── types.py         # Transitions and episode metrics
└── utils/           # Seeding, devices, and evaluation
```

Gymnasium's `terminated` and `truncated` signals are stored separately.
Value-based algorithms, Actor-Critic, A2C, TRPO, and PPO stop bootstrapping only
at true terminal states; REINFORCE updates only after a complete Gymnasium episode. Use
a separate evaluation environment if training will continue afterward, so
evaluation does not disturb the training state.

See the [algorithm documentation index](docs/algorithms.md) for notation,
off-policy and on-policy classifications, and a separate mathematical guide for
every implemented algorithm.

See [ROADMAP.md](ROADMAP.md) for the planned classical RL, DQN, policy-gradient,
continuous-control, offline, model-based, recurrent, and multi-agent phases.
