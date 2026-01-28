# Author: Bradley R. Kinnard
# differential privacy budget accounting with formal guarantees

from dataclasses import dataclass, field
from typing import Any
import math
import time
import json

from utils.helpers import get_logger

logger = get_logger(__name__)


class BudgetExhaustedError(Exception):
    """raised when privacy budget is exhausted."""
    pass


@dataclass
class PrivacySpend:
    """record of a single privacy expenditure."""
    epsilon: float
    delta: float
    mechanism: str
    operation: str
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mechanism": self.mechanism,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class PrivacyBudget:
    """
    privacy budget with hard caps and composition tracking.

    uses advanced composition theorem for tighter bounds:
    for k queries with epsilon each, total epsilon is:
    sqrt(2k * ln(1/delta')) * epsilon + k * epsilon * (e^epsilon - 1)

    simplified: we use basic composition with factor for safety.
    """
    total_epsilon: float
    total_delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    query_count: int = 0
    composition_mode: str = "basic"  # basic or advanced

    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    def remaining_delta(self) -> float:
        return max(0.0, self.total_delta - self.spent_delta)

    def is_exhausted(self) -> bool:
        return self.spent_epsilon >= self.total_epsilon or self.spent_delta >= self.total_delta

    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """check if we can afford this spend."""
        return (
            self.spent_epsilon + epsilon <= self.total_epsilon and
            self.spent_delta + delta <= self.total_delta
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "remaining_epsilon": self.remaining_epsilon(),
            "remaining_delta": self.remaining_delta(),
            "query_count": self.query_count,
            "composition_mode": self.composition_mode,
            "exhausted": self.is_exhausted(),
        }


class PrivacyAccountant:
    """
    formal privacy accounting with audit trail.

    guarantees:
    - budget never exceeds configured limits
    - all spends are logged for audit
    - composition is tracked properly
    - budget exhaustion raises explicit error
    """

    def __init__(
        self,
        total_epsilon: float = 1.0,
        total_delta: float = 1e-5,
        composition_mode: str = "basic",
        audit_logger: Any = None,
    ):
        self._budget = PrivacyBudget(
            total_epsilon=total_epsilon,
            total_delta=total_delta,
            composition_mode=composition_mode,
        )
        self._history: list[PrivacySpend] = []
        self._audit_logger = audit_logger

        logger.info(f"initialized privacy accountant: epsilon={total_epsilon}, delta={total_delta}")

    @property
    def budget(self) -> PrivacyBudget:
        return self._budget

    @property
    def history(self) -> list[PrivacySpend]:
        return self._history.copy()

    def spend(
        self,
        epsilon: float,
        delta: float = 0.0,
        mechanism: str = "laplace",
        operation: str = "unknown",
        details: dict[str, Any] | None = None,
    ) -> PrivacySpend:
        """
        spend privacy budget with formal accounting.

        raises BudgetExhaustedError if budget would be exceeded.
        """
        if epsilon < 0 or delta < 0:
            raise ValueError("epsilon and delta must be non-negative")

        # check if we can afford this
        if not self._budget.can_spend(epsilon, delta):
            msg = (
                f"budget exhausted: requested ({epsilon}, {delta}), "
                f"remaining ({self._budget.remaining_epsilon():.4f}, "
                f"{self._budget.remaining_delta():.6f})"
            )
            logger.error(msg)

            if self._audit_logger:
                self._audit_logger.log(
                    "privacy_budget_exhausted",
                    level="ERROR",
                    details={"requested_epsilon": epsilon, "requested_delta": delta},
                )

            raise BudgetExhaustedError(msg)

        # record the spend
        spend = PrivacySpend(
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism,
            operation=operation,
            timestamp=time.time(),
            details=details or {},
        )

        # update budget
        self._budget.spent_epsilon += epsilon
        self._budget.spent_delta += delta
        self._budget.query_count += 1

        self._history.append(spend)

        logger.debug(
            f"spent ({epsilon}, {delta}) for {operation}, "
            f"remaining ({self._budget.remaining_epsilon():.4f}, "
            f"{self._budget.remaining_delta():.6f})"
        )

        # audit log if available
        if self._audit_logger:
            self._audit_logger.log(
                "privacy_spend",
                level="INFO",
                details=spend.to_dict(),
            )

        return spend

    def compute_epsilon_for_laplace(
        self,
        sensitivity: float,
        scale: float,
    ) -> float:
        """compute epsilon consumed by laplace mechanism."""
        if scale <= 0:
            raise ValueError("scale must be positive")
        return sensitivity / scale

    def compute_epsilon_for_gaussian(
        self,
        sensitivity: float,
        sigma: float,
        delta: float,
    ) -> float:
        """compute epsilon consumed by gaussian mechanism (approximate DP)."""
        if sigma <= 0 or delta <= 0:
            raise ValueError("sigma and delta must be positive")

        # gaussian mechanism: epsilon = sqrt(2 * ln(1.25/delta)) * sensitivity / sigma
        epsilon = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / sigma
        return epsilon

    def get_optimal_noise_scale(
        self,
        sensitivity: float,
        target_epsilon: float,
        mechanism: str = "laplace",
    ) -> float:
        """compute optimal noise scale for target epsilon."""
        if mechanism == "laplace":
            return sensitivity / target_epsilon
        elif mechanism == "gaussian":
            # for gaussian, need delta; use a default
            delta = self._budget.total_delta / 10
            return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / target_epsilon
        else:
            raise ValueError(f"unknown mechanism: {mechanism}")

    def check_invariants(self) -> tuple[bool, str]:
        """verify privacy accounting invariants."""
        # I1: spent never exceeds total
        if self._budget.spent_epsilon > self._budget.total_epsilon:
            return False, "epsilon overspent"
        if self._budget.spent_delta > self._budget.total_delta:
            return False, "delta overspent"

        # I2: history sums to spent
        history_epsilon = sum(s.epsilon for s in self._history)
        history_delta = sum(s.delta for s in self._history)

        if abs(history_epsilon - self._budget.spent_epsilon) > 1e-10:
            return False, "epsilon history mismatch"
        if abs(history_delta - self._budget.spent_delta) > 1e-10:
            return False, "delta history mismatch"

        # I3: query count matches history
        if len(self._history) != self._budget.query_count:
            return False, "query count mismatch"

        return True, "all invariants satisfied"

    def export_audit_report(self) -> dict[str, Any]:
        """export full audit report for external verification."""
        valid, msg = self.check_invariants()

        return {
            "budget": self._budget.to_dict(),
            "history": [s.to_dict() for s in self._history],
            "invariants_valid": valid,
            "invariants_message": msg,
            "exported_at": time.time(),
        }

    def reset(self) -> None:
        """reset budget (for testing only - logs warning)."""
        logger.warning("privacy budget reset - this should only happen in tests")
        self._budget.spent_epsilon = 0.0
        self._budget.spent_delta = 0.0
        self._budget.query_count = 0
        self._history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any], audit_logger: Any = None) -> "PrivacyAccountant":
        """create accountant from config dict."""
        return cls(
            total_epsilon=config.get("total_epsilon", 1.0),
            total_delta=config.get("total_delta", 1e-5),
            composition_mode=config.get("composition_mode", "basic"),
            audit_logger=audit_logger,
        )
