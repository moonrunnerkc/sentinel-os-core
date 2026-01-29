# Author: Bradley R. Kinnard
# formal invariant checker - runtime verification of critical invariants

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

from utils.helpers import get_logger

logger = get_logger(__name__)


class InvariantViolation(Exception):
    """raised when an invariant is violated."""

    def __init__(self, invariant_name: str, message: str, context: dict[str, Any] | None = None):
        self.invariant_name = invariant_name
        self.context = context or {}
        super().__init__(f"invariant '{invariant_name}' violated: {message}")


class CheckResult(Enum):
    """result of an invariant check."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class InvariantCheckResult:
    """result of checking a single invariant."""
    name: str
    result: CheckResult
    message: str
    duration_ms: float
    timestamp: float
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "result": self.result.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "context": self.context,
        }


@dataclass
class VerificationReport:
    """report from a verification run."""
    total_checks: int
    passed: int
    failed: int
    skipped: int
    results: list[InvariantCheckResult]
    duration_ms: float
    timestamp: float
    digest: str = ""

    def __post_init__(self):
        if not self.digest:
            self.digest = self._compute_digest()

    def _compute_digest(self) -> str:
        data = {
            "total": self.total_checks,
            "passed": self.passed,
            "failed": self.failed,
            "results": [r.to_dict() for r in self.results],
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "results": [r.to_dict() for r in self.results],
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "digest": self.digest,
        }

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


class Invariant:
    """
    a named invariant with a check function.

    check function should return (passed: bool, message: str).
    """

    def __init__(
        self,
        name: str,
        check_fn: Callable[[Any], tuple[bool, str]],
        description: str = "",
        critical: bool = True,
    ):
        self._name = name
        self._check_fn = check_fn
        self._description = description
        self._critical = critical

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def critical(self) -> bool:
        return self._critical

    def check(self, state: Any) -> InvariantCheckResult:
        """run the invariant check."""
        start = time.perf_counter()
        try:
            passed, message = self._check_fn(state)
            result = CheckResult.PASSED if passed else CheckResult.FAILED
        except Exception as e:
            result = CheckResult.FAILED
            message = f"check raised exception: {e}"

        duration_ms = (time.perf_counter() - start) * 1000

        return InvariantCheckResult(
            name=self._name,
            result=result,
            message=message,
            duration_ms=duration_ms,
            timestamp=time.time(),
        )


class FormalChecker:
    """
    formal invariant checker.

    registers invariants and checks them against state.
    all checks are logged for audit.
    """

    def __init__(self, fail_fast: bool = True):
        self._invariants: list[Invariant] = []
        self._fail_fast = fail_fast
        self._check_history: list[VerificationReport] = []

    def register(self, invariant: Invariant) -> None:
        """register an invariant."""
        self._invariants.append(invariant)
        logger.debug(f"registered invariant: {invariant.name}")

    def register_fn(
        self,
        name: str,
        check_fn: Callable[[Any], tuple[bool, str]],
        description: str = "",
        critical: bool = True,
    ) -> None:
        """convenience method to register a check function."""
        self.register(Invariant(name, check_fn, description, critical))

    def check_all(self, state: Any) -> VerificationReport:
        """check all invariants against state."""
        start = time.perf_counter()
        results: list[InvariantCheckResult] = []
        passed = 0
        failed = 0
        skipped = 0

        for inv in self._invariants:
            result = inv.check(state)
            results.append(result)

            if result.result == CheckResult.PASSED:
                passed += 1
            elif result.result == CheckResult.FAILED:
                failed += 1
                logger.warning(f"invariant failed: {inv.name} - {result.message}")
                if self._fail_fast and inv.critical:
                    break
            else:
                skipped += 1

        duration_ms = (time.perf_counter() - start) * 1000

        report = VerificationReport(
            total_checks=len(results),
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
            duration_ms=duration_ms,
            timestamp=time.time(),
        )

        self._check_history.append(report)

        if report.all_passed:
            logger.debug(f"all {report.total_checks} invariants passed in {duration_ms:.2f}ms")
        else:
            logger.error(f"verification failed: {failed}/{report.total_checks} invariants failed")

        return report

    def check_and_raise(self, state: Any) -> VerificationReport:
        """check all invariants, raise on failure."""
        report = self.check_all(state)
        if not report.all_passed:
            for r in report.results:
                if r.result == CheckResult.FAILED:
                    raise InvariantViolation(r.name, r.message)
        return report

    def get_history(self) -> list[VerificationReport]:
        """return check history."""
        return self._check_history.copy()

    def clear_history(self) -> None:
        """clear check history."""
        self._check_history = []


# pre-built invariants for common patterns

def belief_confidence_bounded(state: dict[str, Any]) -> tuple[bool, str]:
    """check all belief confidences are in [0, 1]."""
    beliefs = state.get("beliefs", {})
    for bid, belief in beliefs.items():
        conf = belief.get("confidence", 0.0)
        if not (0.0 <= conf <= 1.0):
            return False, f"belief {bid} has confidence {conf} outside [0, 1]"
    return True, f"all {len(beliefs)} beliefs have valid confidence"


def goal_priority_non_negative(state: dict[str, Any]) -> tuple[bool, str]:
    """check all goal priorities are non-negative."""
    goals = state.get("goals", {})
    for gid, goal in goals.items():
        priority = goal.get("priority", 0.0)
        if priority < 0:
            return False, f"goal {gid} has negative priority {priority}"
    return True, f"all {len(goals)} goals have valid priority"


def no_duplicate_belief_ids(state: dict[str, Any]) -> tuple[bool, str]:
    """check no duplicate belief IDs exist."""
    beliefs = state.get("beliefs", {})
    ids = list(beliefs.keys())
    unique_ids = set(ids)
    if len(ids) != len(unique_ids):
        duplicates = [x for x in ids if ids.count(x) > 1]
        return False, f"duplicate belief IDs found: {duplicates[:5]}"
    return True, f"all {len(ids)} belief IDs are unique"


def episode_timestamps_ordered(state: dict[str, Any]) -> tuple[bool, str]:
    """check episodes are ordered by timestamp."""
    episodes = state.get("episodes", [])
    if len(episodes) < 2:
        return True, "insufficient episodes to check ordering"

    for i in range(1, len(episodes)):
        t_prev = episodes[i - 1].get("timestamp", 0)
        t_curr = episodes[i].get("timestamp", 0)
        if t_curr < t_prev:
            return False, f"episode {i} has timestamp {t_curr} < previous {t_prev}"
    return True, f"all {len(episodes)} episodes are ordered by timestamp"


def state_hash_consistent(state: dict[str, Any]) -> tuple[bool, str]:
    """check state hash matches recorded hash."""
    recorded = state.get("_hash", "")
    if not recorded:
        return True, "no hash recorded (skipped)"

    # compute current hash
    state_copy = {k: v for k, v in state.items() if not k.startswith("_")}
    canonical = json.dumps(state_copy, sort_keys=True, separators=(",", ":"), default=str)
    current = hashlib.sha256(canonical.encode()).hexdigest()[:16]

    if current != recorded:
        return False, f"hash mismatch: computed {current} != recorded {recorded}"
    return True, "state hash is consistent"


def memory_within_limits(state: dict[str, Any]) -> tuple[bool, str]:
    """check memory usage is within configured limits."""
    limits = state.get("_limits", {})
    max_beliefs = limits.get("max_beliefs")
    max_episodes = limits.get("max_episodes")

    beliefs = state.get("beliefs", {})
    episodes = state.get("episodes", [])

    if max_beliefs is not None and len(beliefs) > max_beliefs:
        return False, f"belief count {len(beliefs)} exceeds limit {max_beliefs}"
    if max_episodes is not None and len(episodes) > max_episodes:
        return False, f"episode count {len(episodes)} exceeds limit {max_episodes}"

    return True, f"memory within limits (beliefs={len(beliefs)}, episodes={len(episodes)})"


def create_standard_checker(fail_fast: bool = True) -> FormalChecker:
    """create checker with standard invariants."""
    checker = FormalChecker(fail_fast=fail_fast)

    checker.register_fn(
        "belief_confidence_bounded",
        belief_confidence_bounded,
        "all belief confidences must be in [0, 1]",
        critical=True,
    )
    checker.register_fn(
        "goal_priority_non_negative",
        goal_priority_non_negative,
        "all goal priorities must be >= 0",
        critical=True,
    )
    checker.register_fn(
        "no_duplicate_belief_ids",
        no_duplicate_belief_ids,
        "belief IDs must be unique",
        critical=True,
    )
    checker.register_fn(
        "episode_timestamps_ordered",
        episode_timestamps_ordered,
        "episodes must be ordered by timestamp",
        critical=False,
    )
    checker.register_fn(
        "state_hash_consistent",
        state_hash_consistent,
        "state hash must match recorded hash",
        critical=True,
    )
    checker.register_fn(
        "memory_within_limits",
        memory_within_limits,
        "memory usage must be within configured limits",
        critical=False,
    )

    return checker
