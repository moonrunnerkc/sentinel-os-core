# Author: Bradley R. Kinnard
# security module exports

from security.soft_isolation import (
    SoftIsolation,
    IsolationConfig,
    ExecutionResult,
    ThreatModel,
    TrustBoundary,
    SecurityViolation,
    Sandbox,  # backwards compat
)
from security.audit_logger import AuditLogger

__all__ = [
    "SoftIsolation",
    "IsolationConfig",
    "ExecutionResult",
    "ThreatModel",
    "TrustBoundary",
    "SecurityViolation",
    "Sandbox",
    "AuditLogger",
]
