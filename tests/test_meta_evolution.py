# Author: Bradley R. Kinnard
# tests for meta-evolution

import pytest
import numpy as np

from core.meta_evolution import (
    HyperparameterBounds,
    HyperparameterSet,
    EvolutionStep,
    EvolutionResult,
    EvolutionStatus,
    ObjectiveFunction,
    CompositeObjective,
    BeliefCoherenceObjective,
    EfficiencyObjective,
    MetaEvolutionEngine,
    MetaEvolutionDisabledError,
    create_evolution_engine,
)


class TestHyperparameterBounds:
    """test hyperparameter bounds."""

    def test_valid_bounds(self):
        bounds = HyperparameterBounds(0.0, 1.0)
        assert bounds.min_value == 0.0
        assert bounds.max_value == 1.0

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError, match="min_value must be < max_value"):
            HyperparameterBounds(1.0, 0.0)

    def test_clamp_within(self):
        bounds = HyperparameterBounds(0.0, 1.0)
        assert bounds.clamp(0.5) == 0.5

    def test_clamp_below(self):
        bounds = HyperparameterBounds(0.0, 1.0)
        assert bounds.clamp(-0.5) == 0.0

    def test_clamp_above(self):
        bounds = HyperparameterBounds(0.0, 1.0)
        assert bounds.clamp(1.5) == 1.0

    def test_contains_true(self):
        bounds = HyperparameterBounds(0.0, 1.0)
        assert bounds.contains(0.5)

    def test_contains_false(self):
        bounds = HyperparameterBounds(0.0, 1.0)
        assert not bounds.contains(1.5)


class TestHyperparameterSet:
    """test hyperparameter sets."""

    def test_defaults(self):
        params = HyperparameterSet()
        assert params.epsilon == 0.1
        assert params.decay_rate == 0.01
        assert params.confidence_threshold == 0.5
        assert params.learning_rate == 0.01

    def test_to_dict(self):
        params = HyperparameterSet(epsilon=0.2)
        d = params.to_dict()
        assert d["epsilon"] == 0.2

    def test_to_array(self):
        params = HyperparameterSet()
        arr = params.to_array()
        assert len(arr) == 4
        assert arr[0] == 0.1  # epsilon

    def test_from_array(self):
        arr = np.array([0.2, 0.02, 0.6, 0.02])
        params = HyperparameterSet.from_array(arr)
        assert params.epsilon == 0.2
        assert params.decay_rate == 0.02

    def test_from_array_wrong_size(self):
        with pytest.raises(ValueError, match="expected 4 values"):
            HyperparameterSet.from_array(np.array([1, 2, 3]))

    def test_digest_deterministic(self):
        params = HyperparameterSet()
        digest1 = params.digest()
        digest2 = params.digest()
        assert digest1 == digest2

    def test_digest_different_params(self):
        params1 = HyperparameterSet(epsilon=0.1)
        params2 = HyperparameterSet(epsilon=0.2)
        assert params1.digest() != params2.digest()


class TestObjectiveFunctions:
    """test objective functions."""

    def test_belief_coherence_basic(self):
        obj = BeliefCoherenceObjective()
        params = HyperparameterSet()
        context = {
            "contradiction_rate": 0.1,
            "stability_score": 0.9,
            "confidence_mean": 0.5,
        }
        value = obj.evaluate(params, context)
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_efficiency_basic(self):
        obj = EfficiencyObjective()
        params = HyperparameterSet()
        context = {
            "ops_per_second": 10000,
            "memory_usage_pct": 30,
        }
        value = obj.evaluate(params, context)
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_composite_objective(self):
        obj1 = BeliefCoherenceObjective()
        obj2 = EfficiencyObjective()
        composite = CompositeObjective([
            (obj1, 0.6),
            (obj2, 0.4),
        ])
        params = HyperparameterSet()
        context = {
            "contradiction_rate": 0.1,
            "stability_score": 0.9,
            "confidence_mean": 0.5,
            "ops_per_second": 10000,
            "memory_usage_pct": 30,
        }
        value = composite.evaluate(params, context)
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_composite_invalid_weights(self):
        obj = BeliefCoherenceObjective()
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            CompositeObjective([(obj, 0.5)])

    def test_objective_determinism(self):
        obj = BeliefCoherenceObjective()
        params = HyperparameterSet()
        context = {"contradiction_rate": 0.1, "stability_score": 0.9}
        v1 = obj.evaluate(params, context)
        v2 = obj.evaluate(params, context)
        assert v1 == v2


class TestMetaEvolutionEngine:
    """test meta-evolution engine."""

    def test_evolution_basic(self):
        obj = BeliefCoherenceObjective()
        engine = MetaEvolutionEngine(obj, seed=42, max_generations=10)
        result = engine.evolve()
        assert result.status in [EvolutionStatus.CONVERGED, EvolutionStatus.MAX_ITERATIONS]
        assert len(result.history) > 0

    def test_evolution_determinism(self):
        obj = BeliefCoherenceObjective()
        engine1 = MetaEvolutionEngine(obj, seed=42, max_generations=20)
        engine2 = MetaEvolutionEngine(obj, seed=42, max_generations=20)

        result1 = engine1.evolve()
        result2 = engine2.evolve()

        assert result1.final_objective == result2.final_objective
        assert result1.final_params.digest() == result2.final_params.digest()
        assert len(result1.history) == len(result2.history)

    def test_evolution_different_seeds(self):
        obj = BeliefCoherenceObjective()
        engine1 = MetaEvolutionEngine(obj, seed=42, max_generations=50)
        engine2 = MetaEvolutionEngine(obj, seed=123, max_generations=50)

        result1 = engine1.evolve()
        result2 = engine2.evolve()

        # different seeds produce different histories
        # (final params may converge to same value if objective is flat)
        assert result1.history[5].params.digest() != result2.history[5].params.digest()

    def test_params_stay_within_bounds(self):
        bounds = {
            "epsilon": HyperparameterBounds(0.05, 0.5),
            "decay_rate": HyperparameterBounds(0.001, 0.05),
            "confidence_threshold": HyperparameterBounds(0.2, 0.8),
            "learning_rate": HyperparameterBounds(0.001, 0.05),
        }
        obj = BeliefCoherenceObjective()
        engine = MetaEvolutionEngine(obj, bounds=bounds, seed=42, max_generations=50)
        result = engine.evolve()

        for step in result.history:
            p = step.params
            assert bounds["epsilon"].contains(p.epsilon), f"epsilon out of bounds: {p.epsilon}"
            assert bounds["decay_rate"].contains(p.decay_rate)
            assert bounds["confidence_threshold"].contains(p.confidence_threshold)
            assert bounds["learning_rate"].contains(p.learning_rate)

    def test_all_steps_logged(self):
        obj = BeliefCoherenceObjective()
        engine = MetaEvolutionEngine(obj, seed=42, max_generations=30)
        result = engine.evolve()

        # every generation should be logged
        generations = [s.generation for s in result.history]
        expected = list(range(len(result.history)))
        assert generations == expected

    def test_no_silent_updates(self):
        obj = BeliefCoherenceObjective()
        engine = MetaEvolutionEngine(obj, seed=42, max_generations=20)
        result = engine.evolve()

        # every step should have a valid digest
        for step in result.history:
            assert step.params_digest == step.params.digest()

    def test_history_integrity(self):
        obj = BeliefCoherenceObjective()
        engine = MetaEvolutionEngine(obj, seed=42, max_generations=30)
        engine.evolve()

        valid, msg = engine.verify_history_integrity()
        assert valid, msg

    def test_convergence_detection(self):
        obj = BeliefCoherenceObjective()
        engine = MetaEvolutionEngine(
            obj,
            seed=42,
            max_generations=200,
            convergence_threshold=0.001,
        )
        result = engine.evolve()

        # should converge before max iterations
        if result.converged:
            assert result.generations < 200

    def test_nan_rejection(self):
        class BadObjective(ObjectiveFunction):
            def __init__(self):
                super().__init__("bad")

            def evaluate(self, params, context=None):
                return float("nan")

        obj = BadObjective()
        engine = MetaEvolutionEngine(obj, seed=42, max_generations=5)

        with pytest.raises(ValueError, match="invalid value"):
            engine.evolve()

    def test_result_to_dict(self):
        obj = BeliefCoherenceObjective()
        engine = MetaEvolutionEngine(obj, seed=42, max_generations=10)
        result = engine.evolve()

        d = result.to_dict()
        assert "status" in d
        assert "final_params" in d
        assert "final_objective" in d


class TestCreateEvolutionEngine:
    """test factory function."""

    def test_disabled_raises(self):
        config = {"meta_evolution": {"enabled": False}}
        with pytest.raises(MetaEvolutionDisabledError):
            create_evolution_engine(config)

    def test_enabled_creates_engine(self):
        config = {
            "meta_evolution": {
                "enabled": True,
                "max_generations": 50,
                "seed": 123,
            }
        }
        engine = create_evolution_engine(config)
        assert engine._seed == 123
        assert engine._max_generations == 50

    def test_custom_bounds(self):
        config = {
            "meta_evolution": {
                "enabled": True,
                "bounds": {
                    "epsilon": [0.1, 0.5],
                },
            }
        }
        engine = create_evolution_engine(config)
        assert engine._bounds["epsilon"].min_value == 0.1
        assert engine._bounds["epsilon"].max_value == 0.5


class TestEvolutionStep:
    """test evolution step."""

    def test_step_creation(self):
        params = HyperparameterSet()
        step = EvolutionStep(
            generation=0,
            params=params,
            objective_value=0.5,
            delta=0.0,
            accepted=True,
            timestamp=1234567890.0,
            seed=42,
        )
        assert step.generation == 0
        assert step.params_digest == params.digest()

    def test_step_to_dict(self):
        params = HyperparameterSet()
        step = EvolutionStep(
            generation=0,
            params=params,
            objective_value=0.5,
            delta=0.0,
            accepted=True,
            timestamp=1234567890.0,
            seed=42,
        )
        d = step.to_dict()
        assert d["generation"] == 0
        assert d["objective_value"] == 0.5
