# Notebook examples

Use `train_dqn.ipynb` as the reference for style and structure.

- Name notebooks `train_<algorithm>.ipynb`.
- Start with a very brief explanation of the algorithm, environment, and concepts demonstrated.
- Include the algorithm's central mathematical objective or update, briefly defining its symbols.
- Keep imports, environment setup, configuration, and training simple and explicit.
- Use the public `aprenderl` API and an algorithm configuration class when available.
- Plot episode returns and a moving average when applicable.
- End with a rendered policy evaluation using `evaluate_policy`.
- Close environments after use, using `try`/`finally` for evaluation environments.
- Keep prose consistent with the code, including episode counts and behavior.
- Choose settings that train reasonably quickly on CPU.
- Clear outputs and execution counts before committing.
