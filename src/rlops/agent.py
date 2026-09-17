"""PPO agent wrapper over Stable Baselines3 (FR-07, FR-08, FR-09).

``device="cpu"`` is the default because it *is* the measured answer, not a
placeholder left open for tuning: CLAUDE.md §9 benchmarked CPU vs MPS on the
actual training hardware and CPU won by ~12-13x for this policy network
(14.7K parameters — small enough that CPU<->GPU transfer overhead dominates
whatever MPS would otherwise accelerate). Passing ``device="mps"`` here is a
regression, not an optimisation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.rlops.environment import TradingEnvironment


class PPOAgent:
    """Wraps SB3's PPO with this project's environment and conventions."""

    def __init__(
        self,
        env: TradingEnvironment,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.hyperparameters = {
            "policy": "MlpPolicy",
            "learning_rate": learning_rate,
            "gamma": gamma,
            "n_steps": n_steps,
            "seed": seed,
            "device": device,
        }
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            seed=seed,
            device=device,
            verbose=0,
        )

    def train(self, total_timesteps: int) -> "PPOAgent":
        """FR-07: train PPO against the reward `TradingEnvironment` computes."""
        self.model.learn(total_timesteps=total_timesteps)
        return self

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _state = self.model.predict(observation, deterministic=deterministic)
        return action

    def evaluate(
        self,
        env: TradingEnvironment,
        n_episodes: int = 1,
        seed: int | None = None,
        deterministic: bool = True,
    ) -> list[dict[str, np.ndarray]]:
        """Run `n_episodes` rollouts against `env` (the *evaluation*
        partition — walk-forward validation means this is never the same
        environment `train()` was called on).

        Returns the same shape `rlops.baselines.run_policy` does
        (`equity_curve`, `returns` per episode), so an agent's results and a
        baseline's are directly comparable without extra glue (CLAUDE.md
        §10: a Sharpe ratio is meaningless without that comparison).
        """
        results = []
        for i in range(n_episodes):
            episode_seed = None if seed is None else seed + i
            obs, _info = env.reset(seed=episode_seed)
            equity = [env.initial_cash]
            terminated = truncated = False
            while not (terminated or truncated):
                action = self.predict(obs, deterministic=deterministic)
                obs, _reward, terminated, truncated, info = env.step(action)
                equity.append(info["equity"])
            equity_curve = np.asarray(equity, dtype=np.float64)
            returns = np.diff(equity_curve) / equity_curve[:-1]
            results.append({"equity_curve": equity_curve, "returns": returns})
        return results

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))

    @classmethod
    def load(
        cls, path: str | Path, env: TradingEnvironment | None = None, device: str = "cpu"
    ) -> "PPOAgent":
        """FR-09/DR-09: reload a previously registered artifact.

        `env` is optional — a loaded model can `predict()` without one — but
        must be supplied to call `train()` again on the restored agent.
        """
        agent = cls.__new__(cls)
        agent.hyperparameters = {}
        agent.model = PPO.load(str(path), env=env, device=device)
        return agent
