# Study notebooks

Use `04_deep_q_learning/01_dqn.ipynb`,
`06_policy_gradients/01_reinforce.ipynb`, and
`03_temporal_difference/02_q_learning.ipynb` as style references.

## Organization

- Use chapter directories named `<two-digit chapter>_<topic>`.
- Name notebooks `<two-digit lesson>_<topic>.ipynb`.
- Add every notebook to its chapter `README.md`.
- Update `study/README.md` when adding or reordering chapters.
- Preserve and explain prerequisite order.
- Keep each notebook self-contained; do not depend on variables or execution
  from another notebook.

## Content

- Implement algorithms from basic Gymnasium, NumPy, and PyTorch components;
  do not use AprendeRL algorithm implementations.
- Start with the objective and mathematical formulation, defining every symbol.
- Build the implementation in numbered sections.
- Before each major code cell, present the relevant equation, explain its
  terms, and show how they map to the code.
- Keep hyperparameters as named constants and functions small, explicit, and
  educational.
- Explain exploration, bootstrapping, gradient flow, and `terminated` versus
  `truncated` when applicable.
- Plot episode returns with a moving average and other useful training signals.
- End with deterministic or greedy evaluation in a separate rendered
  environment.

## Hygiene

- Close environments and keep prose, equations, and code consistent.
- Choose settings that train reasonably quickly on CPU.
- Clear outputs and execution counts before committing.
