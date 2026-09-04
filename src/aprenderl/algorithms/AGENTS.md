# Algorithm implementations

## File organization

Keep the learning rule easy to find. Prefer this order:

1. Configuration.
2. Construction.
3. Prediction.
4. A clearly labeled learning rule.
5. Environment-interaction hooks.
6. Persistence.
7. Validation and auxiliary helpers.

Small deviations are fine when inheritance or extension hooks make another
order easier to follow.

## Abstraction boundary

Keep algorithmic calculations local, even when short sections are repeated:

- Bellman and TD targets.
- TD errors.
- Policy, value, entropy, and distributional losses.
- Optimizer and gradient-clipping steps.
- Action-value or quantile selection.
- Categorical projections.

Move code into a base class or shared component only when it is substantial,
auxiliary, and repeated across several algorithms. A reader should not need to
jump between files to understand one gradient or value update.

## Inheritance

- Use `OnPolicyAlgorithm` or `OffPolicyAlgorithm` according to the learning
  rule.
- Neural stochastic-policy algorithms should inherit
  `PolicyGradientAlgorithm`.
- DQN variants should reuse DQN lifecycle hooks and override only their network
  construction, value interpretation, target, loss, or replay behavior.
- Tabular algorithms may reuse `TabularValueMixin`, but their TD update must
  remain in the concrete algorithm.

## Environment semantics

Keep `terminated` and `truncated` separate. Bootstrap through truncation unless
the algorithm specifically requires otherwise.

## Checkpoints

- Preserve backward compatibility when changing checkpoint fields.
- Validate checkpoint observation and action spaces against the environment.
- Custom-network checkpoints must require the corresponding network
  architecture when loading.

## Verification

Add focused tests for:

- Target calculations.
- Terminal and truncated transitions.
- Environment-space validation.
- Custom networks.
- Checkpoint round trips.
- Non-zero discrete action-space offsets where applicable.
