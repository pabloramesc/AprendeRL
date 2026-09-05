# AprendeRL study guide

These notebooks form a progressive, book-like introduction to reinforcement
learning. Read the chapters in order; within each chapter, follow the numbered
notebooks.

## Reading order

1. [Getting started](00_getting_started/README.md) — exploration and learning
   from rewards without state transitions.
2. [Dynamic programming](01_dynamic_programming/README.md) — Bellman equations
   when the environment model is known.
3. [Monte Carlo methods](02_monte_carlo/README.md) — learning values and policies
   from complete episodes.
4. [Temporal-difference learning](03_temporal_difference/README.md) —
   bootstrapping, on-policy and off-policy control, traces, and planning.
5. [Deep Q-learning](04_deep_q_learning/README.md) — neural value functions,
   replay buffers, target networks, and DQN improvements.
6. [Distributional reinforcement learning](05_distributional_rl/README.md) —
   categorical and quantile representations of return distributions.
7. [Policy gradients](06_policy_gradients/README.md) — direct policy
   optimization, critics, continuous actions, multi-step targets, TRPO, and PPO.

Each notebook starts from Gymnasium, NumPy, and PyTorch components so the
learning rule remains visible. The `examples/` notebooks demonstrate the
higher-level AprendeRL API instead.
