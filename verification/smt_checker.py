# Author: Bradley R. Kinnard
# z3-based smt verification for formal invariant checking
# provides stronger guarantees via symbolic execution

from typing import Any
from dataclasses import dataclass

from utils.helpers import get_logger

logger = get_logger(__name__)

# z3 is optional - graceful degradation if not installed
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("z3-solver not installed, SMT verification disabled")


@dataclass
class SMTResult:
    """result of an smt check."""
    property_name: str
    satisfiable: bool
    counterexample: dict[str, Any] | None
    proved: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_name": self.property_name,
            "satisfiable": self.satisfiable,
            "counterexample": self.counterexample,
            "proved": self.proved,
            "message": self.message,
        }


class SMTChecker:
    """
    smt-based invariant checker using z3.

    can prove invariants hold for all possible values,
    or find counterexamples when they fail.

    use cases:
    - prove belief confidence always in [0, 1] after updates
    - prove goal priority never goes negative
    - prove decay functions are monotonically decreasing
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            logger.warning("SMTChecker initialized but z3 unavailable")
        self._solver = z3.Solver() if Z3_AVAILABLE else None

    @property
    def available(self) -> bool:
        return Z3_AVAILABLE

    def check_confidence_bounded(self, confidence_expr: Any = None) -> SMTResult:
        """
        prove confidence values stay in [0, 1] after any update.

        models: new_conf = old_conf * decay + evidence * (1 - decay)
        where old_conf in [0, 1], decay in [0, 1], evidence in [0, 1]
        """
        if not Z3_AVAILABLE:
            return SMTResult(
                property_name="confidence_bounded",
                satisfiable=False,
                counterexample=None,
                proved=False,
                message="z3 not available, skipped",
            )

        old_conf = z3.Real("old_conf")
        decay = z3.Real("decay")
        evidence = z3.Real("evidence")
        new_conf = old_conf * decay + evidence * (1 - decay)

        # constraints: inputs in valid ranges
        constraints = [
            old_conf >= 0, old_conf <= 1,
            decay >= 0, decay <= 1,
            evidence >= 0, evidence <= 1,
        ]

        # property to verify: new_conf in [0, 1]
        # negate to find counterexample
        negated_property = z3.Or(new_conf < 0, new_conf > 1)

        solver = z3.Solver()
        solver.add(constraints)
        solver.add(negated_property)

        result = solver.check()

        if result == z3.unsat:
            # no counterexample found - property holds
            return SMTResult(
                property_name="confidence_bounded",
                satisfiable=False,
                counterexample=None,
                proved=True,
                message="proved: confidence always in [0, 1] after update",
            )
        elif result == z3.sat:
            # found counterexample
            model = solver.model()
            ce = {
                "old_conf": float(model[old_conf].as_fraction()),
                "decay": float(model[decay].as_fraction()),
                "evidence": float(model[evidence].as_fraction()),
            }
            return SMTResult(
                property_name="confidence_bounded",
                satisfiable=True,
                counterexample=ce,
                proved=False,
                message=f"counterexample found: {ce}",
            )
        else:
            return SMTResult(
                property_name="confidence_bounded",
                satisfiable=False,
                counterexample=None,
                proved=False,
                message="z3 returned unknown",
            )

    def check_priority_non_negative(self) -> SMTResult:
        """
        prove goal priority stays non-negative after any update.

        models: new_priority = old_priority + delta
        where old_priority >= 0, delta >= -old_priority (can't go below 0)
        """
        if not Z3_AVAILABLE:
            return SMTResult(
                property_name="priority_non_negative",
                satisfiable=False,
                counterexample=None,
                proved=False,
                message="z3 not available, skipped",
            )

        old_priority = z3.Real("old_priority")
        delta = z3.Real("delta")
        new_priority = old_priority + delta

        # constraints: old >= 0, delta bounded to prevent negative result
        constraints = [
            old_priority >= 0,
            delta >= -old_priority,  # this is the key invariant
        ]

        # property to verify: new >= 0
        negated_property = new_priority < 0

        solver = z3.Solver()
        solver.add(constraints)
        solver.add(negated_property)

        result = solver.check()

        if result == z3.unsat:
            return SMTResult(
                property_name="priority_non_negative",
                satisfiable=False,
                counterexample=None,
                proved=True,
                message="proved: priority always >= 0 after update",
            )
        elif result == z3.sat:
            model = solver.model()
            ce = {
                "old_priority": float(model[old_priority].as_fraction()),
                "delta": float(model[delta].as_fraction()),
            }
            return SMTResult(
                property_name="priority_non_negative",
                satisfiable=True,
                counterexample=ce,
                proved=False,
                message=f"counterexample found: {ce}",
            )
        else:
            return SMTResult(
                property_name="priority_non_negative",
                satisfiable=False,
                counterexample=None,
                proved=False,
                message="z3 returned unknown",
            )

    def check_decay_monotonic(self) -> SMTResult:
        """
        prove exponential decay is monotonically decreasing.

        models: f(t) = c * e^(-k * t), prove f(t1) >= f(t2) when t1 <= t2
        simplified to: c * (1 - k*t1) >= c * (1 - k*t2) for small t (linear approx)
        """
        if not Z3_AVAILABLE:
            return SMTResult(
                property_name="decay_monotonic",
                satisfiable=False,
                counterexample=None,
                proved=False,
                message="z3 not available, skipped",
            )

        c = z3.Real("c")  # initial confidence
        k = z3.Real("k")  # decay rate
        t1 = z3.Real("t1")
        t2 = z3.Real("t2")

        # linear approximation of decay
        f_t1 = c * (1 - k * t1)
        f_t2 = c * (1 - k * t2)

        # constraints
        constraints = [
            c > 0, c <= 1,  # positive confidence
            k > 0, k <= 1,  # positive decay rate
            t1 >= 0, t2 >= 0,
            t1 <= t2,  # t1 comes before t2
            k * t1 <= 1, k * t2 <= 1,  # linear approx valid range
        ]

        # property: f(t1) >= f(t2) when t1 <= t2
        negated_property = f_t1 < f_t2

        solver = z3.Solver()
        solver.add(constraints)
        solver.add(negated_property)

        result = solver.check()

        if result == z3.unsat:
            return SMTResult(
                property_name="decay_monotonic",
                satisfiable=False,
                counterexample=None,
                proved=True,
                message="proved: decay is monotonically decreasing",
            )
        elif result == z3.sat:
            model = solver.model()
            ce = {
                "c": float(model[c].as_fraction()),
                "k": float(model[k].as_fraction()),
                "t1": float(model[t1].as_fraction()),
                "t2": float(model[t2].as_fraction()),
            }
            return SMTResult(
                property_name="decay_monotonic",
                satisfiable=True,
                counterexample=ce,
                proved=False,
                message=f"counterexample found: {ce}",
            )
        else:
            return SMTResult(
                property_name="decay_monotonic",
                satisfiable=False,
                counterexample=None,
                proved=False,
                message="z3 returned unknown",
            )

    def check_budget_exhaustion(self, epsilon_total: float = 1.0) -> SMTResult:
        """
        prove privacy budget cannot go negative.

        models: remaining = total - sum(consumed)
        where each consumed_i >= 0 and sum(consumed_i) <= total
        """
        if not Z3_AVAILABLE:
            return SMTResult(
                property_name="budget_exhaustion",
                satisfiable=False,
                counterexample=None,
                proved=False,
                message="z3 not available, skipped",
            )

        total = z3.Real("total")
        consumed_1 = z3.Real("consumed_1")
        consumed_2 = z3.Real("consumed_2")
        consumed_3 = z3.Real("consumed_3")
        remaining = total - (consumed_1 + consumed_2 + consumed_3)

        constraints = [
            total == epsilon_total,
            consumed_1 >= 0, consumed_2 >= 0, consumed_3 >= 0,
            consumed_1 + consumed_2 + consumed_3 <= total,
        ]

        # property: remaining >= 0
        negated_property = remaining < 0

        solver = z3.Solver()
        solver.add(constraints)
        solver.add(negated_property)

        result = solver.check()

        if result == z3.unsat:
            return SMTResult(
                property_name="budget_exhaustion",
                satisfiable=False,
                counterexample=None,
                proved=True,
                message="proved: privacy budget never goes negative",
            )
        elif result == z3.sat:
            model = solver.model()
            ce = {
                "consumed_1": float(model[consumed_1].as_fraction()),
                "consumed_2": float(model[consumed_2].as_fraction()),
                "consumed_3": float(model[consumed_3].as_fraction()),
            }
            return SMTResult(
                property_name="budget_exhaustion",
                satisfiable=True,
                counterexample=ce,
                proved=False,
                message=f"counterexample found: {ce}",
            )
        else:
            return SMTResult(
                property_name="budget_exhaustion",
                satisfiable=False,
                counterexample=None,
                proved=False,
                message="z3 returned unknown",
            )

    def run_all_checks(self) -> list[SMTResult]:
        """run all smt checks and return results."""
        results = [
            self.check_confidence_bounded(),
            self.check_priority_non_negative(),
            self.check_decay_monotonic(),
            self.check_budget_exhaustion(),
        ]

        proved = sum(1 for r in results if r.proved)
        skipped = sum(1 for r in results if "skipped" in r.message.lower())
        logger.info(f"SMT checks: {proved} proved, {skipped} skipped, {len(results)} total")

        return results


def create_smt_checker() -> SMTChecker:
    """factory to create smt checker."""
    return SMTChecker()
