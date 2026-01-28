# Author: Bradley R. Kinnard
# property-based testing for formal verification refinement

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Generator
import random
import time

from verification.state_machine import (
    SystemState,
    BeliefState,
    GoalState,
    Contradiction,
    TransitionEngine,
)
from verification.invariants import InvariantChecker, check_contradiction_closure
from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class PropertyResult:
    """result of a property test run."""
    property_name: str
    passed: bool
    iterations: int
    failures: list[dict[str, Any]]
    seed: int
    elapsed_ms: float


def property_test(
    name: str,
    iterations: int = 100,
    seed: int | None = None,
):
    """decorator for property-based tests."""
    def decorator(fn: Callable[..., bool]):
        def wrapper(*args, **kwargs) -> PropertyResult:
            actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
            random.seed(actual_seed)

            failures = []
            start = time.time()

            for i in range(iterations):
                try:
                    result = fn(*args, iteration=i, **kwargs)
                    if not result:
                        failures.append({"iteration": i, "result": "returned False"})
                except Exception as e:
                    failures.append({"iteration": i, "exception": str(e)})

            elapsed = (time.time() - start) * 1000

            return PropertyResult(
                property_name=name,
                passed=len(failures) == 0,
                iterations=iterations,
                failures=failures[:10],  # keep first 10
                seed=actual_seed,
                elapsed_ms=elapsed,
            )
        return wrapper
    return decorator


class PropertyTester:
    """
    property-based testing framework for sentinel os.

    tests properties that should hold for all valid inputs,
    serving as executable refinement of formal specs.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)
        self._invariant_checker = InvariantChecker()

    def _random_belief(self, belief_id: str | None = None, timestamp: float | None = None) -> BeliefState:
        """generate random valid belief."""
        bid = belief_id or f"belief_{self._rng.randint(0, 10000)}"
        content = f"content_{self._rng.randint(0, 10000)}"
        # use provided timestamp or generate deterministic one from rng
        ts = timestamp if timestamp is not None else float(self._rng.randint(0, 1000000))
        return BeliefState(
            belief_id=bid,
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            confidence=self._rng.random(),
            timestamp=ts,
            source="property_test",
        )

    def _random_goal(self, goal_id: str | None = None) -> GoalState:
        """generate random valid goal."""
        gid = goal_id or f"goal_{self._rng.randint(0, 10000)}"
        content = f"goal_content_{self._rng.randint(0, 10000)}"
        return GoalState(
            goal_id=gid,
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            priority=self._rng.random(),
            status=self._rng.choice(["active", "collapsed", "abandoned"]),
        )

    def test_belief_insert_preserves_invariants(self, iterations: int = 100) -> PropertyResult:
        """property: inserting valid beliefs never violates invariants."""
        failures = []
        start = time.time()

        for i in range(iterations):
            engine = TransitionEngine()

            # insert random number of beliefs
            n = self._rng.randint(1, 50)
            for j in range(n):
                belief = self._random_belief(f"b_{i}_{j}")
                engine.insert_belief(belief)

            # check invariants
            violations = self._invariant_checker.check_all(engine.state)
            if violations:
                failures.append({
                    "iteration": i,
                    "violations": [v.message for v in violations],
                })

        elapsed = (time.time() - start) * 1000

        return PropertyResult(
            property_name="belief_insert_preserves_invariants",
            passed=len(failures) == 0,
            iterations=iterations,
            failures=failures[:10],
            seed=self._seed,
            elapsed_ms=elapsed,
        )

    def test_determinism_under_seed(self, iterations: int = 50) -> PropertyResult:
        """property: same operations with same seed yield same state digest."""
        failures = []
        start = time.time()

        for i in range(iterations):
            # run same sequence twice with same seed
            digests = []

            for run in range(2):
                self._rng = random.Random(self._seed + i)
                engine = TransitionEngine(SystemState(seed=self._seed + i))

                n = self._rng.randint(5, 20)
                for j in range(n):
                    belief = self._random_belief(f"det_{j}")
                    engine.insert_belief(belief)

                digests.append(engine.state.digest())

            if digests[0] != digests[1]:
                failures.append({
                    "iteration": i,
                    "digest_a": digests[0][:16],
                    "digest_b": digests[1][:16],
                })

        elapsed = (time.time() - start) * 1000

        return PropertyResult(
            property_name="determinism_under_seed",
            passed=len(failures) == 0,
            iterations=iterations,
            failures=failures[:10],
            seed=self._seed,
            elapsed_ms=elapsed,
        )

    def test_trace_integrity(self, iterations: int = 50) -> PropertyResult:
        """property: transition trace chains correctly."""
        failures = []
        start = time.time()

        for i in range(iterations):
            engine = TransitionEngine()

            n = self._rng.randint(10, 30)
            for j in range(n):
                belief = self._random_belief(f"trace_{i}_{j}")
                engine.insert_belief(belief)

            valid, msg = engine.verify_trace_integrity()
            if not valid:
                failures.append({
                    "iteration": i,
                    "message": msg,
                })

        elapsed = (time.time() - start) * 1000

        return PropertyResult(
            property_name="trace_integrity",
            passed=len(failures) == 0,
            iterations=iterations,
            failures=failures[:10],
            seed=self._seed,
            elapsed_ms=elapsed,
        )

    def test_confidence_bounds_after_update(self, iterations: int = 100) -> PropertyResult:
        """property: confidence remains bounded after any update."""
        failures = []
        start = time.time()

        for i in range(iterations):
            engine = TransitionEngine()

            # insert some beliefs
            for j in range(5):
                engine.insert_belief(self._random_belief(f"bound_{i}_{j}"))

            # try extreme updates
            for bid in list(engine.state.beliefs.keys()):
                extreme = self._rng.choice([-10.0, 0.0, 0.5, 1.0, 10.0])
                engine.update_belief(bid, extreme, time.time())

            # check all confidences are bounded
            for bid, belief in engine.state.beliefs.items():
                if not 0.0 <= belief.confidence <= 1.0:
                    failures.append({
                        "iteration": i,
                        "belief_id": bid,
                        "confidence": belief.confidence,
                    })

        elapsed = (time.time() - start) * 1000

        return PropertyResult(
            property_name="confidence_bounds_after_update",
            passed=len(failures) == 0,
            iterations=iterations,
            failures=failures[:10],
            seed=self._seed,
            elapsed_ms=elapsed,
        )

    def run_all_properties(self, iterations: int = 50) -> list[PropertyResult]:
        """run all property tests."""
        results = [
            self.test_belief_insert_preserves_invariants(iterations),
            self.test_determinism_under_seed(iterations),
            self.test_trace_integrity(iterations),
            self.test_confidence_bounds_after_update(iterations),
        ]

        passed = sum(1 for r in results if r.passed)
        logger.info(f"property tests: {passed}/{len(results)} passed")

        return results


def generate_random_traces(
    count: int = 10,
    max_ops: int = 100,
    seed: int = 42,
) -> Generator[list[dict[str, Any]], None, None]:
    """generate random operation traces for testing."""
    rng = random.Random(seed)

    for _ in range(count):
        trace = []
        n = rng.randint(10, max_ops)

        for i in range(n):
            op_type = rng.choice(["insert_belief", "update_belief"])

            if op_type == "insert_belief":
                trace.append({
                    "op": "insert_belief",
                    "belief_id": f"b_{i}",
                    "confidence": rng.random(),
                })
            elif op_type == "update_belief" and trace:
                # pick a previous belief to update
                prev = rng.choice([t for t in trace if t["op"] == "insert_belief"])
                trace.append({
                    "op": "update_belief",
                    "belief_id": prev["belief_id"],
                    "new_confidence": rng.random(),
                })

        yield trace
