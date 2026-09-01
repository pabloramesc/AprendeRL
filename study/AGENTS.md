# Study notebooks

Use `dqn.ipynb`, `reinforce.ipynb`, and `q_learning.ipynb` as references for style and structure.

- Name notebooks `<algorithm>.ipynb`.
- Implement the algorithm from basic Gymnasium, NumPy, and PyTorch components instead of using AprendeRL abstractions.
- Start with the algorithm's objective and mathematical formulation, defining every symbol used.
- Build the implementation in numbered sections. Before each major code cell, present the relevant equation, explain its terms, and show how they map to the code.
- Keep hyperparameters as named constants and functions small, explicit, and educational.
- Explain important implementation details such as exploration, bootstrapping, gradient flow, and `terminated` versus `truncated` when applicable.
- Plot episode returns with a moving average and other useful training signals such as loss.
- End with a deterministic or greedy evaluation in a separate rendered environment.
- Close environments and keep prose, equations, and code consistent.
- Clear outputs and execution counts before committing.
