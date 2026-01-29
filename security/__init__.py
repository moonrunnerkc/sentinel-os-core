# Author: Bradley R. Kinnard
# security module exports

from security.isolation import (
    Isolation,
    IsolationLevel,
    IsolationConfig,
    IsolationUnavailableError,
    ExecutionResult,
    TrustBoundary,
    SecurityViolation,
    LEVEL_THREAT_MODELS,
    get_level_threat_model,
    check_level_available,
    # backwards compatibility
    SoftIsolation,
    Sandbox,
)
from security.audit_logger import AuditLogger

__all__ = [
    # new exports
    "Isolation",
    "IsolationLevel",
    "IsolationConfig",
    "IsolationUnavailableError",
    "ExecutionResult",
    "TrustBoundary",
    "SecurityViolation",
    "LEVEL_THREAT_MODELS",
    "get_level_threat_model",
    "check_level_available",
    "AuditLogger",
    # backwards compatibility
    "SoftIsolation",
    "Sandbox",
]
