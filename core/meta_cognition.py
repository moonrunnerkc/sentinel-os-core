# Author: Bradley R. Kinnard
# meta cognition - offline meta-RL for hyperparameter evolution

from typing import Any

import numpy as np

from utils.helpers import get_logger


logger = get_logger(__name__)


class MetaCognition:
    """
    offline meta-RL for evolving hyperparameters from introspective logs.
    no code modifications - only hyperparameter tuning.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._hyperparams: dict[str, float] = {
            "epsilon": 0.1,           # differential privacy
            "decay_rate": 0.01,       # belief decay
            "learning_rate": 0.05,    # rl update rate
            "exploration": 0.2,       # exploration vs exploitation
            "convergence_threshold": 1e-5
        }
        self._history: list[dict[str, Any]] = []

    def get_hyperparams(self) -> dict[str, float]:
        """return current hyperparameters."""
        return self._hyperparams.copy()

    def set_hyperparam(self, name: str, value: float) -> None:
        """set a specific hyperparameter."""
        if name not in self._hyperparams:
            raise KeyError(f"unknown hyperparameter: {name}")
        self._hyperparams[name] = value
        logger.info(f"set {name} = {value}")

    def parse_introspection_logs(
        self,
        logs: list[dict[str, Any]]
    ) -> dict[str, float]:
        """parse logs to extract performance metrics."""
        if not logs:
            return {}

        metrics = {
            "avg_convergence_time": 0.0,
            "success_rate": 0.0,
            "avg_reward": 0.0
        }

        convergence_times = []
        successes = 0
        rewards = []

        for entry in logs:
            if "convergence_iterations" in entry:
                convergence_times.append(entry["convergence_iterations"])
            if entry.get("converged", False):
                successes += 1
            if "reward" in entry:
                rewards.append(entry["reward"])

        if convergence_times:
            metrics["avg_convergence_time"] = np.mean(convergence_times)
        if logs:
            metrics["success_rate"] = successes / len(logs)
        if rewards:
            metrics["avg_reward"] = np.mean(rewards)

        logger.debug(f"parsed metrics: {metrics}")
        return metrics

    def meta_rl_evolve(
        self,
        introspection_logs: list[dict[str, Any]],
        iterations: int = 1000,
        seed: int | None = None
    ) -> dict[str, float]:
        """
        evolve hyperparameters using offline meta-RL.
        seeded, logged, inspectable.
        """
        if seed is None:
            seed = self._seed
        np.random.seed(seed)

        logger.info(f"starting meta-RL evolution: {iterations} iterations, seed={seed}")

        metrics = self.parse_introspection_logs(introspection_logs)
        best_hyperparams = self._hyperparams.copy()
        best_score = self._evaluate_hyperparams(metrics)

        for i in range(iterations):
            # propose mutation
            candidate = self._mutate_hyperparams(best_hyperparams)

            # evaluate (simulated - in practice would run actual trials)
            score = self._evaluate_hyperparams(metrics, candidate)

            if score > best_score:
                best_score = score
                best_hyperparams = candidate
                logger.debug(f"iteration {i}: new best score {score:.4f}")

            # check for convergence
            if i > 0 and abs(score - best_score) < self._hyperparams["convergence_threshold"]:
                logger.info(f"meta-RL converged at iteration {i}")
                break

        self._hyperparams = best_hyperparams
        self._history.append({
            "iterations": iterations,
            "seed": seed,
            "final_score": best_score,
            "hyperparams": best_hyperparams.copy()
        })

        logger.info(f"meta-RL complete: final hyperparams = {best_hyperparams}")
        return best_hyperparams

    def _mutate_hyperparams(
        self,
        params: dict[str, float]
    ) -> dict[str, float]:
        """mutate hyperparameters with gaussian noise."""
        mutated = params.copy()

        # small gaussian perturbation
        for key in mutated:
            noise = np.random.normal(0, 0.01)
            mutated[key] = max(1e-6, mutated[key] + noise)

        return mutated

    def _evaluate_hyperparams(
        self,
        metrics: dict[str, float],
        params: dict[str, float] | None = None
    ) -> float:
        """evaluate hyperparameter configuration."""
        if params is None:
            params = self._hyperparams

        # composite score based on metrics
        score = 0.0

        # faster convergence is better (inverse)
        if metrics.get("avg_convergence_time", 0) > 0:
            score += 1.0 / metrics["avg_convergence_time"]

        # higher success rate is better
        score += metrics.get("success_rate", 0) * 10

        # higher reward is better
        score += metrics.get("avg_reward", 0)

        # penalize extreme hyperparams
        for value in params.values():
            if value > 1.0 or value < 1e-6:
                score -= 0.1

        return score

    def get_evolution_history(self) -> list[dict[str, Any]]:
        """return history of meta-RL runs."""
        return self._history.copy()
