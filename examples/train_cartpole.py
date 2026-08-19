"""Train and evaluate Double DQN on Gymnasium CartPole."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym

from aprenderl import DoubleDQN, DoubleDQNConfig
from aprenderl.config import ExperimentConfig
from aprenderl.logging import TrainingLogger
from aprenderl.utils import evaluate_policy


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/double_dqn_cartpole.pt"),
    )
    return parser.parse_args()


def main() -> None:
    """Train, evaluate, and save a CartPole agent."""

    args = parse_args()
    experiment = ExperimentConfig(
        total_timesteps=args.timesteps,
        evaluation_episodes=args.eval_episodes,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    config = DoubleDQNConfig(
        learning_rate=1e-3,
        buffer_size=20_000,
        learning_starts=1_000,
        target_update_interval=500,
        exploration_fraction=0.25,
        seed=experiment.seed,
    )

    try:
        agent = DoubleDQN(
            train_env,
            config,
            device=experiment.device,
            logger=TrainingLogger(verbose=True),
        )
        agent.learn(experiment.total_timesteps)
        evaluation = evaluate_policy(
            agent,
            eval_env,
            episodes=experiment.evaluation_episodes,
            seed=experiment.seed + 1_000,
        )
        agent.save(experiment.checkpoint_path)
    finally:
        train_env.close()
        eval_env.close()

    recent = agent.episode_returns[-10:]
    recent_mean = sum(recent) / len(recent) if recent else 0.0
    print("Algorithm: Double DQN")
    print(f"Timesteps: {agent.num_timesteps}")
    print(f"Updates: {agent.num_updates}")
    print(f"Mean return (last {len(recent)} training episodes): {recent_mean:.1f}")
    print(
        f"Mean return ({experiment.evaluation_episodes} evaluation episodes): "
        f"{evaluation.mean_return:.1f} ± {evaluation.return_std:.1f}"
    )
    print(f"Checkpoint: {experiment.checkpoint_path}")


if __name__ == "__main__":
    main()
