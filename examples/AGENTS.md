# Notebook examples

Use `dqn.ipynb` as the reference for discrete value-based examples and
`reinforce_continuous.ipynb` for continuous-policy examples.

## Purpose and organization

- Demonstrate the public `aprenderl` API; keep algorithm reimplementations in
  the study notebooks.
- Name notebooks `<algorithm>.ipynb`.
- Update README and algorithm-documentation links when renaming an example.

## Content

- Start with a brief explanation of the algorithm, environment, and concepts.
- Include the central mathematical objective or update and briefly define its
  symbols.
- Keep imports, environment setup, configuration, and training explicit.
- Use an algorithm configuration class when available.
- Plot episode returns and a moving average when applicable.
- End with rendered policy evaluation using `evaluate_policy` and a separate
  evaluation environment.
- Close environments with `try`/`finally` and keep prose consistent with code.
- Choose settings that train reasonably quickly on CPU.

## Hygiene

- Do not commit execution counts, cell outputs, checkpoints, videos, or other
  generated artifacts.
