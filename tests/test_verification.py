# Author: Bradley R. Kinnard
# tests for formal verification module

import pytest
import time
import hashlib

from verification.state_machine import (
    BeliefState,
    GoalState,
    Contradiction,
    SystemState,
    StateTransition,
    TransitionType,
    TransitionEngine,
)
from verification.invariants import (
    InvariantChecker,
    InvariantViolation,
    InvariantSeverity,
    check_all_invariants,
    check_contradiction_closure,
    check_determinism,
    ContractionGuarantee,
)
from verification.properties import PropertyTester


class TestBeliefState:
    """tests for belief state model."""

    def test_create_valid_belief(self):
        belief = BeliefState(
            belief_id="b1",
            content_hash="abc123",
            confidence=0.5,
            timestamp=time.time(),
        )
        assert belief.belief_id == "b1"
        assert belief.confidence == 0.5

    def test_confidence_bounds(self):
        # valid bounds
        BeliefState("b1", "h", 0.0, 0.0)
        BeliefState("b1", "h", 1.0, 0.0)

        # invalid bounds
        with pytest.raises(ValueError):
            BeliefState("b1", "h", -0.1, 0.0)
        with pytest.raises(ValueError):
            BeliefState("b1", "h", 1.1, 0.0)

    def test_immutability(self):
        belief = BeliefState("b1", "h", 0.5, 0.0)
        with pytest.raises(AttributeError):
            belief.confidence = 0.6

    def test_serialization(self):
        belief = BeliefState("b1", "h", 0.5, 123.456)
        d = belief.to_dict()
        restored = BeliefState.from_dict(d)
        assert belief == restored


class TestGoalState:
    """tests for goal state model."""

    def test_create_valid_goal(self):
        goal = GoalState(
            goal_id="g1",
            content_hash="xyz",
            priority=0.8,
            status="active",
        )
        assert goal.status == "active"

    def test_invalid_status(self):
        with pytest.raises(ValueError):
            GoalState("g1", "h", 0.5, "invalid_status")

    def test_valid_statuses(self):
        for status in ["active", "collapsed", "abandoned"]:
            GoalState("g1", "h", 0.5, status)


class TestSystemState:
    """tests for system state."""

    def test_empty_state(self):
        state = SystemState()
        assert len(state.beliefs) == 0
        assert len(state.goals) == 0
        assert state.step_counter == 0

    def test_digest_deterministic(self):
        state = SystemState(seed=42)
        state.beliefs["b1"] = BeliefState("b1", "h", 0.5, 0.0)

        d1 = state.digest()
        d2 = state.digest()
        assert d1 == d2

    def test_digest_changes_with_state(self):
        state = SystemState()
        d1 = state.digest()

        state.beliefs["b1"] = BeliefState("b1", "h", 0.5, 0.0)
        d2 = state.digest()

        assert d1 != d2


class TestTransitionEngine:
    """tests for transition engine."""

    def test_insert_belief(self):
        engine = TransitionEngine()
        belief = BeliefState("b1", "h", 0.5, time.time())

        transition = engine.insert_belief(belief)

        assert "b1" in engine.state.beliefs
        assert transition.transition_type == TransitionType.BELIEF_INSERT
        assert engine.state.step_counter == 1

    def test_update_belief(self):
        engine = TransitionEngine()
        engine.insert_belief(BeliefState("b1", "h", 0.5, time.time()))

        transition = engine.update_belief("b1", 0.8, time.time())

        assert engine.state.beliefs["b1"].confidence == 0.8
        assert transition.transition_type == TransitionType.BELIEF_UPDATE

    def test_update_clamps_confidence(self):
        engine = TransitionEngine()
        engine.insert_belief(BeliefState("b1", "h", 0.5, time.time()))

        engine.update_belief("b1", 1.5, time.time())
        assert engine.state.beliefs["b1"].confidence == 1.0

        engine.update_belief("b1", -0.5, time.time())
        assert engine.state.beliefs["b1"].confidence == 0.0

    def test_trace_integrity(self):
        engine = TransitionEngine()

        for i in range(10):
            engine.insert_belief(BeliefState(f"b{i}", "h", 0.5, time.time()))

        valid, msg = engine.verify_trace_integrity()
        assert valid, msg

    def test_duplicate_insert_fails(self):
        engine = TransitionEngine()
        engine.insert_belief(BeliefState("b1", "h", 0.5, time.time()))

        with pytest.raises(ValueError):
            engine.insert_belief(BeliefState("b1", "h", 0.6, time.time()))


class TestInvariantChecker:
    """tests for invariant checking."""

    def test_valid_state_passes(self):
        state = SystemState()
        state.beliefs["b1"] = BeliefState("b1", "h", 0.5, 0.0)

        violations = check_all_invariants(state)
        assert len(violations) == 0

    def test_negative_step_counter_fails(self):
        state = SystemState()
        state.step_counter = -1

        violations = check_all_invariants(state)
        assert any(v.invariant_name == "I5_step_monotonic" for v in violations)

    def test_contradiction_closure(self):
        state = SystemState()
        state.beliefs["b1"] = BeliefState("b1", "h", 0.5, 0.0)
        state.beliefs["b2"] = BeliefState("b2", "h", 0.6, 0.0)
        state.contradictions["c1"] = Contradiction(
            "c1", "b1", "b2", 0.0, resolved=False
        )

        valid, msg = check_contradiction_closure(state, max_unresolved=0)
        assert not valid

        valid, msg = check_contradiction_closure(state, max_unresolved=1)
        assert valid


class TestPropertyTester:
    """tests for property-based testing framework."""

    def test_belief_insert_preserves_invariants(self):
        tester = PropertyTester(seed=42)
        result = tester.test_belief_insert_preserves_invariants(iterations=50)
        assert result.passed, f"failures: {result.failures}"

    def test_determinism_under_seed(self):
        tester = PropertyTester(seed=42)
        result = tester.test_determinism_under_seed(iterations=20)
        assert result.passed, f"failures: {result.failures}"

    def test_trace_integrity(self):
        tester = PropertyTester(seed=42)
        result = tester.test_trace_integrity(iterations=20)
        assert result.passed, f"failures: {result.failures}"

    def test_confidence_bounds_after_update(self):
        tester = PropertyTester(seed=42)
        result = tester.test_confidence_bounds_after_update(iterations=50)
        assert result.passed, f"failures: {result.failures}"


class TestContractionGuarantee:
    """tests for formal termination guarantees."""

    def test_max_steps_formula(self):
        # n*(n-1)/2 for n beliefs
        assert ContractionGuarantee.max_steps(0) == 0
        assert ContractionGuarantee.max_steps(1) == 0
        assert ContractionGuarantee.max_steps(2) == 1
        assert ContractionGuarantee.max_steps(10) == 45
        assert ContractionGuarantee.max_steps(100) == 4950

    def test_termination_check(self):
        state = SystemState()
        for i in range(10):
            state.beliefs[f"b{i}"] = BeliefState(f"b{i}", "h", 0.5, 0.0)

        valid, _ = ContractionGuarantee.check_termination(state, steps_taken=40)
        assert valid

        valid, _ = ContractionGuarantee.check_termination(state, steps_taken=50)
        assert not valid
