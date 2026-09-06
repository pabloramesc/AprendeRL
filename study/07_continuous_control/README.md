# Chapter 7: Continuous control and soft actor-critic

## Learning objectives

- Differentiate a critic through a deterministic action.
- Reuse off-policy experience with replay and target networks.
- Understand TD3's twin critics, target smoothing, and delayed actor updates.
- Reparameterize bounded Gaussian actions and correct their log densities.
- Optimize maximum-entropy returns and tune the entropy temperature.
- Replace sampled action expectations with exact categorical sums.

## Prerequisites

Read [deep Q-learning](../04_deep_q_learning/README.md) for replay and target
networks, then [policy gradients](../06_policy_gradients/README.md) for actors,
critics, continuous actions, and gradient flow.

## Notebooks

1. [DPG: differentiate through the action](01_dpg.ipynb)
2. [DDPG: deep networks, replay and target networks](02_ddpg.ipynb)
3. [TD3: twin critics, smoothing and delayed updates](03_td3.ipynb)
4. [SAC: reparameterization and entropy tuning](04_sac.ipynb)
5. [Discrete SAC: exact action expectations](05_discrete_sac.ipynb)

DPG uses an analytic one-step tracking task to isolate the gradient. DDPG and
TD3 extend it to long-horizon Pendulum control. SAC replaces external
exploration noise with a learned stochastic policy; discrete SAC transfers
that objective to CartPole. Each notebook is self-contained and implements
its update directly with Gymnasium, NumPy, and PyTorch.

The default runs use CPU, explicit seeds, return plots, and deterministic
rendered evaluation in a separate environment. Short runs illustrate learning;
performance varies across seeds and tasks. Use `RENDER_MODE = "rgb_array"`
for headless execution. The [public API examples](../../README.md#example-notebooks)
show the corresponding library classes; DPG remains a study-only concept.
