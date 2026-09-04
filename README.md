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
- REINFORCE, Actor-Critic, and A2C for discrete and continuous actions.

Every algorithm follows `learn`, `predict`, `save`, and `load`. Smaller
foundational methods and individual DQN mechanisms remain as self-contained
study notebooks rather than additional public classes.

## Example notebooks

Complete training examples:

- [Tabular Q-Learning](examples/train_qlearning.ipynb)
- [Tabular SARSA](examples/train_sarsa.ipynb)
- [DQN with Double DQN targets](examples/train_dqn.ipynb)
- [C51](examples/train_c51.ipynb)
- [Rainbow DQN](examples/train_rainbow_dqn.ipynb)
- [QR-DQN](examples/train_qr_dqn.ipynb)
- [IQN](examples/train_iqn.ipynb)
- [FQF](examples/train_fqf.ipynb)
- [REINFORCE](examples/train_reinforce.ipynb)
- [REINFORCE on continuous actions](examples/train_reinforce_continuous.ipynb)
- [Actor-Critic](examples/train_actor_critic.ipynb)
- [Actor-Critic on continuous actions](examples/train_actor_critic_continuous.ipynb)
- [A2C](examples/train_a2c.ipynb)
- [A2C on continuous actions](examples/train_a2c_continuous.ipynb)

Each notebook defines its Gymnasium environment with an `ENV_ID` constant near
the beginning, so you can switch to another environment compatible with the
algorithm's observation and action spaces.

## Study notebooks

These notebooks implement concepts directly with Gymnasium, NumPy, and
PyTorch. They intentionally cover more algorithms and intermediate variants
than the maintained public API.

- [Multi-Armed Bandits](study/multi_armed_bandit.ipynb)
- [Value Iteration](study/value_iteration.ipynb)
- [Policy Iteration](study/policy_iteration.ipynb)
- [Monte Carlo Prediction](study/monte_carlo_prediction.ipynb)
- [Monte Carlo Control](study/monte_carlo_control.ipynb)
- [Tabular Q-Learning](study/q_learning.ipynb)
- [Tabular SARSA](study/sarsa.ipynb)
- [Expected SARSA](study/expected_sarsa.ipynb)
- [n-step SARSA](study/n_step_sarsa.ipynb)
- [SARSA(lambda)](study/sarsa_lambda.ipynb)
- [Dyna-Q](study/dyna_q.ipynb)
- [DQN](study/dqn.ipynb)
- [Double DQN](study/double_dqn.ipynb)
- [Dueling DQN](study/dueling_dqn.ipynb)
- [Prioritized DQN](study/prioritized_dqn.ipynb)
- [n-step DQN](study/n_step_dqn.ipynb)
- [NoisyNet DQN](study/noisy_dqn.ipynb)
- [C51](study/c51.ipynb)
- [Rainbow DQN](study/rainbow_dqn.ipynb)
- [QR-DQN](study/qr_dqn.ipynb)
- [IQN](study/iqn.ipynb)
- [REINFORCE](study/reinforce.ipynb)
- [REINFORCE on continuous actions](study/reinforce_continuous.ipynb)
- [Actor-Critic](study/actor_critic.ipynb)
- [Actor-Critic on continuous actions](study/actor_critic_continuous.ipynb)

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
Value-based algorithms, Actor-Critic, and A2C stop bootstrapping only at true
terminal states; REINFORCE updates only after a complete Gymnasium episode. Use
a separate evaluation environment if training will continue afterward, so
evaluation does not disturb the training state.

See the [algorithm documentation index](docs/algorithms.md) for notation,
off-policy and on-policy classifications, and a separate mathematical guide for
every implemented algorithm.

See [ROADMAP.md](ROADMAP.md) for the planned classical RL, DQN, policy-gradient,
continuous-control, offline, model-based, recurrent, and multi-agent phases.
