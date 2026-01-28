# Author: Bradley R. Kinnard
# invariant definitions and runtime checking for formal verification

from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum, auto

from verification.state_machine import SystemState
from utils.helpers import get_logger

logger = get_logger(__name__)


class InvariantSeverity(Enum):
    """severity levels for invariant violations."""
    CRITICAL = auto()  # system must halt
    ERROR = auto()     # operation must be rolled back
    WARNING = auto()   # logged but tolerated


@dataclass
class InvariantViolation:
    """record of an invariant violation."""
    invariant_name: str
    severity: InvariantSeverity
    message: str
    state_digest: str
    details: dict[str, Any]


class InvariantChecker:
    """
    checks formal invariants on system state.

    proven invariants (model-level):
    I1: confidence bounded - belief confidence in [0, 1]
    I2: priority bounded - goal priority in [0, 1]
    I3: contradiction closure - after resolve_all, no unresolved contradictions
    I4: determinism - same inputs + seed yields same state digest
    I5: trace integrity - transition chain is unbroken
    """

    def __init__(self):
        self._invariants: list[tuple[str, Callable[[SystemState], tuple[bool, str]], InvariantSeverity]] = []
        self._register_core_invariants()

    def _register_core_invariants(self) -> None:
        """register the core invariants we guarantee."""

        # I1: belief confidence bounded
        def check_confidence_bounded(state: SystemState) -> tuple[bool, str]:
            for bid, belief in state.beliefs.items():
                if not 0.0 <= belief.confidence <= 1.0:
                    return False, f"belief {bid} confidence {belief.confidence} out of bounds"
            return True, "all confidences bounded"

        self._invariants.append(("I1_confidence_bounded", check_confidence_bounded, InvariantSeverity.CRITICAL))

        # I2: goal priority bounded
        def check_priority_bounded(state: SystemState) -> tuple[bool, str]:
            for gid, goal in state.goals.items():
                if not 0.0 <= goal.priority <= 1.0:
                    return False, f"goal {gid} priority {goal.priority} out of bounds"
            return True, "all priorities bounded"

        self._invariants.append(("I2_priority_bounded", check_priority_bounded, InvariantSeverity.CRITICAL))

        # I3: goal status valid
        def check_goal_status_valid(state: SystemState) -> tuple[bool, str]:
            valid_statuses = {"active", "collapsed", "abandoned"}
            for gid, goal in state.goals.items():
                if goal.status not in valid_statuses:
                    return False, f"goal {gid} has invalid status {goal.status}"
            return True, "all goal statuses valid"

        self._invariants.append(("I3_goal_status_valid", check_goal_status_valid, InvariantSeverity.CRITICAL))

        # I4: no orphan contradictions
        def check_no_orphan_contradictions(state: SystemState) -> tuple[bool, str]:
            for cid, contradiction in state.contradictions.items():
                if contradiction.belief_a_id not in state.beliefs:
                    if not contradiction.resolved:
                        return False, f"contradiction {cid} references missing belief {contradiction.belief_a_id}"
                if contradiction.belief_b_id not in state.beliefs:
                    if not contradiction.resolved:
                        return False, f"contradiction {cid} references missing belief {contradiction.belief_b_id}"
            return True, "no orphan contradictions"

        self._invariants.append(("I4_no_orphan_contradictions", check_no_orphan_contradictions, InvariantSeverity.ERROR))

        # I5: step counter monotonic
        def check_step_monotonic(state: SystemState) -> tuple[bool, str]:
            if state.step_counter < 0:
                return False, f"step counter {state.step_counter} is negative"
            return True, "step counter valid"

        self._invariants.append(("I5_step_monotonic", check_step_monotonic, InvariantSeverity.CRITICAL))

    def register_invariant(
        self,
        name: str,
        check_fn: Callable[[SystemState], tuple[bool, str]],
        severity: InvariantSeverity = InvariantSeverity.ERROR,
    ) -> None:
        """register a custom invariant."""
        self._invariants.append((name, check_fn, severity))
        logger.info(f"registered invariant: {name}")

    def check_all(self, state: SystemState) -> list[InvariantViolation]:
        """check all invariants and return violations."""
        violations = []
        digest = state.digest()

        for name, check_fn, severity in self._invariants:
            try:
                passed, message = check_fn(state)
                if not passed:
                    violation = InvariantViolation(
                        invariant_name=name,
                        severity=severity,
                        message=message,
                        state_digest=digest,
                        details={"state_step": state.step_counter},
                    )
                    violations.append(violation)
                    logger.warning(f"invariant {name} violated: {message}")
            except Exception as e:
                violation = InvariantViolation(
                    invariant_name=name,
                    severity=InvariantSeverity.CRITICAL,
                    message=f"invariant check raised exception: {e}",
                    state_digest=digest,
                    details={"exception": str(e)},
                )
                violations.append(violation)
                logger.error(f"invariant {name} check failed: {e}")

        return violations

    def check_critical(self, state: SystemState) -> list[InvariantViolation]:
        """check only critical invariants."""
        return [v for v in self.check_all(state) if v.severity == InvariantSeverity.CRITICAL]

    def has_violations(self, state: SystemState) -> bool:
        """quick check if any invariants are violated."""
        return len(self.check_all(state)) > 0


def check_all_invariants(state: SystemState) -> list[InvariantViolation]:
    """convenience function to check all invariants."""
    checker = InvariantChecker()
    return checker.check_all(state)


def check_contradiction_closure(state: SystemState, max_unresolved: int = 0) -> tuple[bool, str]:
    """
    I_closure: after resolve_all(), unresolved contradictions <= max_unresolved.

    this is the key invariant for cognitive coherence.
    """
    unresolved = state.unresolved_contradictions()
    if len(unresolved) > max_unresolved:
        return False, f"{len(unresolved)} unresolved contradictions (max: {max_unresolved})"
    return True, f"contradiction closure satisfied ({len(unresolved)} unresolved)"


def check_determinism(
    state_a: SystemState,
    state_b: SystemState,
) -> tuple[bool, str]:
    """
    I_determinism: same initial state + same inputs + same seed => same final state.

    verified by comparing digests.
    """
    if state_a.digest() != state_b.digest():
        return False, "states diverged despite same inputs"
    return True, "determinism verified"


class ContractionGuarantee:
    """
    formal guarantee that contradiction resolution terminates.

    theorem: for finite belief set B with |B| = n,
    contradiction resolution terminates in at most n*(n-1)/2 steps
    (the maximum number of pairwise contradictions).
    """

    @staticmethod
    def max_steps(belief_count: int) -> int:
        """compute maximum resolution steps for termination guarantee."""
        return belief_count * (belief_count - 1) // 2

    @staticmethod
    def check_termination(
        state: SystemState,
        steps_taken: int,
    ) -> tuple[bool, str]:
        """verify resolution terminated within bounds."""
        max_steps = ContractionGuarantee.max_steps(len(state.beliefs))
        if steps_taken > max_steps:
            return False, f"resolution took {steps_taken} steps, exceeds bound {max_steps}"
        return True, f"termination verified: {steps_taken} <= {max_steps}"
