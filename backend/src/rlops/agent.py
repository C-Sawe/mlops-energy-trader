"""RL agent wrappers over Stable Baselines3 (FR-07, FR-08, FR-09).

``device="cpu"`` is the default because it *is* the measured answer, not a
placeholder left open for tuning: CLAUDE.md §9 benchmarked CPU vs MPS on the
actual training hardware and CPU won by ~12-13x for this policy network
(14.7K parameters — small enough that CPU<->GPU transfer overhead dominates
whatever MPS would otherwise accelerate). Passing ``device="mps"`` here is a
regression, not an optimisation.

`PPOAgent` is what the production CT loop trains and serves. `RLAgent`
generalises it to the other continuous-action algorithms SB3 ships, for the
algorithm-comparison study (`scripts/algorithm_comparison.py`); each runs at
its library defaults, the same "untuned" footing PPO has always had here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from src.rlops.environment import TradingEnvironment

ALGORITHMS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3}

# SB3's TD3 defaults to no exploration noise, which leaves a deterministic
# policy with nothing driving exploration; 0.1 is the value SB3's own TD3
# examples use.
_TD3_NOISE_SIGMA = 0.1


class RLAgent:
    """Wraps one SB3 algorithm with this project's environment and conventions."""

    def __init__(
        self,
        env: TradingEnvironment,
        algorithm: str = "ppo",
        seed: int | None = None,
        device: str = "cpu",
        **algo_kwargs,
    ) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {sorted(ALGORITHMS)}")
        self.algorithm = algorithm
        self.hyperparameters = {
            "algorithm": algorithm,
            "policy": "MlpPolicy",
            **algo_kwargs,
            "seed": seed,
            "device": device,
        }
        if algorithm == "td3" and "action_noise" not in algo_kwargs:
            n = env.action_space.shape[0]
            algo_kwargs["action_noise"] = NormalActionNoise(np.zeros(n), _TD3_NOISE_SIGMA * np.ones(n))
            self.hyperparameters["action_noise_sigma"] = _TD3_NOISE_SIGMA
        self.model = ALGORITHMS[algorithm](
            "MlpPolicy", env, seed=seed, device=device, verbose=0, **algo_kwargs
        )

    def train(self, total_timesteps: int) -> "RLAgent":
        """FR-07: train against the reward `TradingEnvironment` computes."""
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
        cls,
        path: str | Path,
        env: TradingEnvironment | None = None,
        device: str = "cpu",
        algorithm: str = "ppo",
    ) -> "RLAgent":
        """FR-09/DR-09: reload a previously registered artifact.

        `env` is optional — a loaded model can `predict()` without one — but
        must be supplied to call `train()` again on the restored agent.
        """
        agent = cls.__new__(cls)
        agent.algorithm = algorithm
        agent.hyperparameters = {}
        agent.model = ALGORITHMS[algorithm].load(str(path), env=env, device=device)
        return agent


class PPOAgent(RLAgent):
    """The production agent: PPO, as named in the proposal's Section 3.6."""

    def __init__(
        self,
        env: TradingEnvironment,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            env,
            algorithm="ppo",
            seed=seed,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
        )

    @classmethod
    def load(
        cls, path: str | Path, env: TradingEnvironment | None = None, device: str = "cpu"
    ) -> "PPOAgent":
        return super().load(path, env=env, device=device, algorithm="ppo")
