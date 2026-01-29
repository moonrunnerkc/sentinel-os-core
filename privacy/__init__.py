# Author: Bradley R. Kinnard
# privacy module - differential privacy budget accounting and enforcement

from privacy.budget import (
    PrivacyBudget,
    PrivacyAccountant,
    BudgetExhaustedError,
    PrivacySpend,
)
from privacy.mechanisms import (
    laplace_mechanism,
    gaussian_mechanism,
    clip_l2_norm,
)


# singleton enforcer for global privacy budget
_global_accountant: PrivacyAccountant | None = None


def get_privacy_enforcer(
    total_epsilon: float = 1.0,
    total_delta: float = 1e-5,
) -> PrivacyAccountant:
    """
    get global privacy enforcer singleton.

    creates a new accountant on first call or if budget is exhausted.
    all privacy-sensitive operations should use this.
    """
    global _global_accountant

    if _global_accountant is None:
        _global_accountant = PrivacyAccountant(
            total_epsilon=total_epsilon,
            total_delta=total_delta,
        )

    return _global_accountant


def reset_privacy_enforcer() -> None:
    """reset global enforcer. use only for testing."""
    global _global_accountant
    _global_accountant = None


__all__ = [
    "PrivacyBudget",
    "PrivacyAccountant",
    "BudgetExhaustedError",
    "PrivacySpend",
    "laplace_mechanism",
    "gaussian_mechanism",
    "clip_l2_norm",
    "get_privacy_enforcer",
    "reset_privacy_enforcer",
]
