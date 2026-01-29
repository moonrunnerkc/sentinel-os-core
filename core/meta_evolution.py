# Author: Bradley R. Kinnard
# meta-evolution - hyperparameter optimization with auditable, deterministic evolution

import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable
from enum import Enum

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


class EvolutionStatus(Enum):
    """status of evolution run."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"


@dataclass
class HyperparameterBounds:
    """bounds for a hyperparameter."""
    min_value: float
    max_value: float

    def __post_init__(self):
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value must be < max_value: {self.min_value} >= {self.max_value}")

    def clamp(self, value: float) -> float:
        """clamp value to bounds."""
        return max(self.min_value, min(self.max_value, value))

    def contains(self, value: float) -> bool:
        """check if value is within bounds."""
        return self.min_value <= value <= self.max_value


@dataclass
class HyperparameterSet:
    """set of hyperparameters for evolution."""
    epsilon: float = 0.1
    decay_rate: float = 0.01
    confidence_threshold: float = 0.5
    learning_rate: float = 0.01

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.epsilon,
            self.decay_rate,
            self.confidence_threshold,
            self.learning_rate,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "HyperparameterSet":
        if len(arr) != 4:
            raise ValueError(f"expected 4 values, got {len(arr)}")
        return cls(
            epsilon=float(arr[0]),
            decay_rate=float(arr[1]),
            confidence_threshold=float(arr[2]),
            learning_rate=float(arr[3]),
        )

    def digest(self) -> str:
        """compute deterministic hash of params."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class EvolutionStep:
    """single step in evolution history."""
    generation: int
    params: HyperparameterSet
    objective_value: float
    delta: float
    accepted: bool
    timestamp: float
    seed: int
    params_digest: str = field(default="")

    def __post_init__(self):
        if not self.params_digest:
            self.params_digest = self.params.digest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "params": self.params.to_dict(),
            "objective_value": self.objective_value,
            "delta": self.delta,
            "accepted": self.accepted,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "params_digest": self.params_digest,
        }


@dataclass
class EvolutionResult:
    """result of evolution run."""
    status: EvolutionStatus
    final_params: HyperparameterSet
    final_objective: float
    generations: int
    history: list[EvolutionStep]
    seed: int
    converged: bool
    convergence_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "final_params": self.final_params.to_dict(),
            "final_objective": self.final_objective,
            "generations": self.generations,
            "history_length": len(self.history),
            "seed": self.seed,
            "converged": self.converged,
            "convergence_delta": self.convergence_delta,
        }


class ObjectiveFunction:
    """
    objective function for meta-evolution.

    must be deterministic and measurable.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        params: HyperparameterSet,
        context: dict[str, Any] | None = None,
    ) -> float:
        """evaluate objective. must return finite float."""
        raise NotImplementedError

    def validate_result(self, value: float) -> bool:
        """validate result is finite and usable."""
        return np.isfinite(value)


class CompositeObjective(ObjectiveFunction):
    """
    composite objective combining multiple objectives with weights.

    objective = sum(weight_i * objective_i)
    """

    def __init__(
        self,
        objectives: list[tuple[ObjectiveFunction, float]],
        name: str = "composite",
    ):
        super().__init__(name)
        self._objectives = objectives
        total_weight = sum(w for _, w in objectives)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"weights must sum to 1.0, got {total_weight}")

    def evaluate(
        self,
        params: HyperparameterSet,
        context: dict[str, Any] | None = None,
    ) -> float:
        total = 0.0
        for obj, weight in self._objectives:
            value = obj.evaluate(params, context)
            if not self.validate_result(value):
                raise ValueError(f"objective {obj.name} returned invalid value: {value}")
            total += weight * value
        return total


class BeliefCoherenceObjective(ObjectiveFunction):
    """
    objective measuring belief system coherence.

    higher = better (fewer contradictions, more stable beliefs)
    """

    def __init__(self):
        super().__init__("belief_coherence")

    def evaluate(
        self,
        params: HyperparameterSet,
        context: dict[str, Any] | None = None,
    ) -> float:
        context = context or {}

        # get belief stats from context
        contradiction_rate = context.get("contradiction_rate", 0.0)
        stability_score = context.get("stability_score", 1.0)
        confidence_mean = context.get("confidence_mean", 0.5)

        # coherence is inverse of contradictions, scaled by stability
        coherence = (1.0 - contradiction_rate) * stability_score

        # penalize extreme epsilon (too noisy or too rigid)
        epsilon_penalty = abs(params.epsilon - 0.1) * 0.1

        # reward confidence threshold alignment with mean
        threshold_bonus = 1.0 - abs(params.confidence_threshold - confidence_mean) * 0.2

        return coherence - epsilon_penalty + threshold_bonus * 0.1


class EfficiencyObjective(ObjectiveFunction):
    """
    objective measuring system efficiency.

    higher = better (faster operations, lower memory)
    """

    def __init__(self):
        super().__init__("efficiency")

    def evaluate(
        self,
        params: HyperparameterSet,
        context: dict[str, Any] | None = None,
    ) -> float:
        context = context or {}

        # get efficiency metrics from context
        ops_per_second = context.get("ops_per_second", 1000.0)
        memory_usage_pct = context.get("memory_usage_pct", 50.0)

        # normalize ops/sec (higher is better, log scale)
        ops_score = np.log10(max(1.0, ops_per_second)) / 6.0  # normalize to ~1 at 1M ops

        # memory score (lower is better)
        memory_score = 1.0 - (memory_usage_pct / 100.0)

        # penalize high learning rates (cause instability)
        lr_penalty = max(0, params.learning_rate - 0.05) * 2.0

        return (ops_score + memory_score) / 2.0 - lr_penalty


class MetaEvolutionEngine:
    """
    meta-evolution engine for hyperparameter optimization.

    uses hill-climbing with bounded mutations.
    fully deterministic under fixed seed.
    all steps logged for audit.
    """

    def __init__(
        self,
        objective: ObjectiveFunction,
        bounds: dict[str, HyperparameterBounds] | None = None,
        seed: int = 42,
        max_generations: int = 100,
        convergence_threshold: float = 0.001,
        mutation_scale: float = 0.1,
    ):
        self._objective = objective
        self._bounds = bounds or self._default_bounds()
        self._seed = seed
        self._max_generations = max_generations
        self._convergence_threshold = convergence_threshold
        self._mutation_scale = mutation_scale

        self._rng = np.random.default_rng(seed)
        self._history: list[EvolutionStep] = []
        self._status = EvolutionStatus.NOT_STARTED
        self._current_params: HyperparameterSet | None = None
        self._current_objective: float | None = None

    @staticmethod
    def _default_bounds() -> dict[str, HyperparameterBounds]:
        return {
            "epsilon": HyperparameterBounds(0.01, 1.0),
            "decay_rate": HyperparameterBounds(0.001, 0.1),
            "confidence_threshold": HyperparameterBounds(0.1, 0.9),
            "learning_rate": HyperparameterBounds(0.001, 0.1),
        }

    def _clamp_params(self, params: HyperparameterSet) -> HyperparameterSet:
        """clamp all params to bounds."""
        return HyperparameterSet(
            epsilon=self._bounds["epsilon"].clamp(params.epsilon),
            decay_rate=self._bounds["decay_rate"].clamp(params.decay_rate),
            confidence_threshold=self._bounds["confidence_threshold"].clamp(params.confidence_threshold),
            learning_rate=self._bounds["learning_rate"].clamp(params.learning_rate),
        )

    def _validate_params(self, params: HyperparameterSet) -> bool:
        """check all params are within bounds."""
        return (
            self._bounds["epsilon"].contains(params.epsilon)
            and self._bounds["decay_rate"].contains(params.decay_rate)
            and self._bounds["confidence_threshold"].contains(params.confidence_threshold)
            and self._bounds["learning_rate"].contains(params.learning_rate)
        )

    def _mutate(self, params: HyperparameterSet) -> HyperparameterSet:
        """generate mutated candidate."""
        arr = params.to_array()
        # add gaussian noise scaled by mutation_scale
        noise = self._rng.normal(0, self._mutation_scale, size=arr.shape)
        mutated = arr + noise * arr  # proportional mutation
        candidate = HyperparameterSet.from_array(mutated)
        return self._clamp_params(candidate)

    def _evaluate(
        self,
        params: HyperparameterSet,
        context: dict[str, Any] | None = None,
    ) -> float:
        """evaluate objective, validate result."""
        value = self._objective.evaluate(params, context)
        if not self._objective.validate_result(value):
            raise ValueError(f"objective returned invalid value: {value}")
        return value

    def _log_step(
        self,
        generation: int,
        params: HyperparameterSet,
        objective_value: float,
        delta: float,
        accepted: bool,
    ) -> EvolutionStep:
        """log evolution step to history."""
        step = EvolutionStep(
            generation=generation,
            params=params,
            objective_value=objective_value,
            delta=delta,
            accepted=accepted,
            timestamp=time.time(),
            seed=self._seed,
        )
        self._history.append(step)
        logger.debug(
            f"gen {generation}: obj={objective_value:.6f} delta={delta:.6f} accepted={accepted}"
        )
        return step

    def evolve(
        self,
        initial_params: HyperparameterSet | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvolutionResult:
        """
        run evolution to optimize hyperparameters.

        fully deterministic under fixed seed.
        all steps logged.
        """
        # reset RNG for reproducibility
        self._rng = np.random.default_rng(self._seed)
        self._history = []
        self._status = EvolutionStatus.RUNNING

        # initialize
        if initial_params is None:
            initial_params = HyperparameterSet()
        current = self._clamp_params(initial_params)
        current_obj = self._evaluate(current, context)

        self._log_step(0, current, current_obj, 0.0, True)
        logger.info(f"meta-evolution started: seed={self._seed}, initial_obj={current_obj:.6f}")

        best = current
        best_obj = current_obj
        converged = False
        last_improvement_gen = 0

        for gen in range(1, self._max_generations + 1):
            # generate candidate
            candidate = self._mutate(current)
            candidate_obj = self._evaluate(candidate, context)

            delta = candidate_obj - current_obj
            accepted = delta > 0  # maximize objective

            self._log_step(gen, candidate, candidate_obj, delta, accepted)

            if accepted:
                current = candidate
                current_obj = candidate_obj

                if current_obj > best_obj:
                    best = current
                    best_obj = current_obj
                    last_improvement_gen = gen

            # check convergence (no improvement for N generations)
            if gen - last_improvement_gen > 10 and gen > 20:
                recent_deltas = [
                    abs(s.delta) for s in self._history[-10:]
                ]
                if max(recent_deltas) < self._convergence_threshold:
                    converged = True
                    self._status = EvolutionStatus.CONVERGED
                    logger.info(f"meta-evolution converged at generation {gen}")
                    break

        if not converged:
            self._status = EvolutionStatus.MAX_ITERATIONS
            logger.info(f"meta-evolution reached max generations ({self._max_generations})")

        self._current_params = best
        self._current_objective = best_obj

        result = EvolutionResult(
            status=self._status,
            final_params=best,
            final_objective=best_obj,
            generations=len(self._history),
            history=self._history.copy(),
            seed=self._seed,
            converged=converged,
            convergence_delta=self._convergence_threshold,
        )

        logger.info(
            f"meta-evolution complete: status={self._status.value}, "
            f"final_obj={best_obj:.6f}, generations={len(self._history)}"
        )

        return result

    def get_history(self) -> list[EvolutionStep]:
        """return evolution history for audit."""
        return self._history.copy()

    def verify_history_integrity(self) -> tuple[bool, str]:
        """verify history has not been tampered with."""
        if not self._history:
            return True, "empty history"

        for i, step in enumerate(self._history):
            expected_digest = step.params.digest()
            if step.params_digest != expected_digest:
                return False, f"digest mismatch at step {i}"

            if i > 0:
                if step.generation != self._history[i - 1].generation + 1:
                    return False, f"generation gap at step {i}"

        return True, f"history valid ({len(self._history)} steps)"


class MetaEvolutionDisabledError(Exception):
    """raised when meta-evolution is called but disabled."""
    pass


def create_evolution_engine(
    config: dict[str, Any],
    objective: ObjectiveFunction | None = None,
) -> MetaEvolutionEngine:
    """
    create evolution engine from config.

    raises MetaEvolutionDisabledError if disabled in config.
    """
    meta_config = config.get("meta_evolution", {})

    if not meta_config.get("enabled", False):
        raise MetaEvolutionDisabledError(
            "meta-evolution is disabled in config. set meta_evolution.enabled=true to use."
        )

    # parse bounds from config
    bounds_config = meta_config.get("bounds", {})
    bounds = {}
    for param in ["epsilon", "decay_rate", "confidence_threshold", "learning_rate"]:
        if param in bounds_config:
            b = bounds_config[param]
            bounds[param] = HyperparameterBounds(b[0], b[1])

    if objective is None:
        objective = BeliefCoherenceObjective()

    return MetaEvolutionEngine(
        objective=objective,
        bounds=bounds if bounds else None,
        seed=meta_config.get("seed", 42),
        max_generations=meta_config.get("max_generations", 100),
        convergence_threshold=meta_config.get("convergence_threshold", 0.001),
    )
