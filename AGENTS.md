# AprendeRL

AprendeRL is an educational reinforcement-learning library with a simple, SB3-inspired API.

## Architecture

- `BaseAlgorithm` is the common base class.
- `OnPolicyAlgorithm` and `OffPolicyAlgorithm` derive from it.
- Policies and network architectures are separate from algorithms.

## Conventions

- Keep implementations simple, explicit, and educational.
- Follow Gymnasium and Stable-Baselines3 conventions where practical.
- New algorithms should include documentation, tests, and example and study notebooks.

## API design

- Algorithms should support:
  - `__init__(policy, env, ...)`
  - `learn(total_timesteps, ...)`
  - `predict(observation, deterministic=False)`
  - `save(path)`
  - `load(path, env=None)`
- Use SB3 terminology for common parameters such as `learning_rate`, `gamma`, `batch_size`, `buffer_size`, `train_freq`, and `gradient_steps`.
- Prefer SB3-like public interfaces without reproducing its internal complexity.

## Post-implementation

- After each implementation, review the code for opportunities to simplify or refactor it while preserving clarity and consistency with the overall architecture.
- Run and fix tests until the expected behavior is verified.
- Verify that README files and documentation remain aligned with the current code and public API.
