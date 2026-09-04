# AprendeRL

AprendeRL is an educational reinforcement-learning library with a compact,
SB3-inspired API.

## Architecture

- `BaseAlgorithm` owns environment interaction, callbacks, logging, and common
  training state.
- `OnPolicyAlgorithm` and `OffPolicyAlgorithm` identify the learning family.
- `PolicyGradientAlgorithm` contains shared neural-policy infrastructure.
- `TabularValueMixin` contains shared tabular infrastructure.
- `DQN` is the common foundation for DQN-family algorithms.
- Policies, networks, buffers, and algorithms remain separate components.

## Design principles

- Keep implementations explicit, readable, and educational.
- Keep each algorithm's mathematical update in its concrete module.
- Do not extract short algorithmic calculations solely to remove duplication.
- Share substantial auxiliary plumbing when it clearly reduces noise.
- Follow Gymnasium and Stable-Baselines3 terminology where practical without
  reproducing their internal complexity.
- Treat `terminated` and `truncated` according to Gymnasium semantics.

## Public API

Algorithms should support:

- `__init__(env, ..., config=None)`
- `learn(total_timesteps, ...)`
- `predict(observation, deterministic=False)`
- `save(path)`
- `load(path, env, ...)`

Use familiar parameter names such as `learning_rate`, `gamma`, `batch_size`,
`buffer_size`, `train_freq`, and `gradient_steps`.

## New algorithms

New public algorithms should include:

- Unit tests for their central update.
- Checkpoint and environment-validation tests.
- Algorithm documentation.
- A public-API example notebook.
- A study notebook when the algorithm introduces a new educational concept.

## Post-implementation

- Simplify only when readability improves.
- Run Ruff and the complete test suite.
- Verify README and documentation links.
- Preserve the public API and checkpoint compatibility unless explicitly
  changing them.
