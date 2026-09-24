"""Tests for the PPO agent wrapper (FR-07, FR-08, FR-09).

Uses tiny hyperparameters (small `n_steps`, few timesteps) purely to keep
the suite fast — none of this asserts anything about learned behaviour,
only that the wrapper's mechanics (train/predict/save/load/evaluate) work
and that a reloaded agent is bit-for-bit the same policy.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.rlops.agent import ALGORITHMS, PPOAgent, RLAgent
from tests.test_environment import _feature_frame
from src.rlops.environment import TradingEnvironment


def _make_env(tickers=("XOM", "CVX")) -> TradingEnvironment:
    frame = _feature_frame(n=250, tickers=tickers)
    return TradingEnvironment(frame, tickers=tickers)


def test_agent_uses_cpu_by_default():
    """CLAUDE.md §9: measured ~12-13x faster than mps on this hardware."""
    agent = PPOAgent(_make_env(), n_steps=64)
    assert str(agent.model.device) == "cpu"


def test_agent_records_hyperparameters_for_later_logging():
    agent = PPOAgent(_make_env(), learning_rate=1e-3, gamma=0.95, n_steps=64, seed=7)
    assert agent.hyperparameters["learning_rate"] == 1e-3
    assert agent.hyperparameters["gamma"] == 0.95
    assert agent.hyperparameters["n_steps"] == 64
    assert agent.hyperparameters["seed"] == 7


def test_agent_trains_without_error():
    agent = PPOAgent(_make_env(), n_steps=64, seed=0)
    agent.train(total_timesteps=128)  # no exception is the assertion


def test_agent_predict_returns_action_within_bounds():
    env = _make_env()
    agent = PPOAgent(env, n_steps=64, seed=0)
    obs, _ = env.reset()

    action = agent.predict(obs)

    assert action.shape == env.action_space.shape
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_agent_evaluate_matches_baseline_result_shape():
    """CLAUDE.md §10: an agent's results must be directly comparable to a
    baseline's, which means the same shape as `baselines.run_policy`."""
    train_env = _make_env()
    eval_env = _make_env()
    agent = PPOAgent(train_env, n_steps=64, seed=0).train(total_timesteps=128)

    results = agent.evaluate(eval_env, n_episodes=2, seed=1)

    assert len(results) == 2
    for result in results:
        assert set(result.keys()) == {"equity_curve", "returns"}
        assert result["equity_curve"][0] == pytest.approx(eval_env.initial_cash)
        assert len(result["returns"]) == len(result["equity_curve"]) - 1


def test_agent_save_and_load_roundtrip_predicts_identically(tmp_path):
    env = _make_env()
    agent = PPOAgent(env, n_steps=64, seed=0)
    obs, _ = env.reset()

    before = agent.predict(obs, deterministic=True)

    path = tmp_path / "agent.zip"
    agent.save(path)
    reloaded = PPOAgent.load(path)
    after = reloaded.predict(obs, deterministic=True)

    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_every_comparison_algorithm_trains_and_acts_within_bounds(algorithm):
    env = _make_env()
    agent = RLAgent(env, algorithm=algorithm, seed=0).train(total_timesteps=200)
    obs, _ = env.reset()

    action = agent.predict(obs)

    assert action.shape == env.action_space.shape
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert agent.hyperparameters["algorithm"] == algorithm


def test_unknown_algorithm_is_rejected_not_defaulted():
    with pytest.raises(ValueError, match="unknown algorithm"):
        RLAgent(_make_env(), algorithm="dqn")


def test_td3_gets_exploration_noise_by_default():
    """SB3's TD3 has no action noise unless given one, so an untuned TD3
    would never explore; the wrapper must supply it."""
    agent = RLAgent(_make_env(), algorithm="td3", seed=0)
    assert agent.model.action_noise is not None
    assert agent.hyperparameters["action_noise_sigma"] == 0.1


def test_non_ppo_save_and_load_roundtrip_predicts_identically(tmp_path):
    env = _make_env()
    agent = RLAgent(env, algorithm="sac", seed=0)
    obs, _ = env.reset()
    before = agent.predict(obs, deterministic=True)

    path = tmp_path / "sac.zip"
    agent.save(path)
    after = RLAgent.load(path, algorithm="sac").predict(obs, deterministic=True)

    np.testing.assert_array_equal(before, after)
