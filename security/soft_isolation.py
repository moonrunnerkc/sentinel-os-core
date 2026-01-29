# Author: Bradley R. Kinnard
# soft isolation - explicit threat model, honest about limitations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import signal
import threading
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

from utils.helpers import get_logger

logger = get_logger(__name__)


class ThreatModel(Enum):
    """
    explicit threat model for soft isolation.

    what this sandbox defends against:
    - ACCIDENTAL: typos, basic mistakes, unintended operations
    - OPPORTUNISTIC: script-kiddie injection, obvious exploits

    what this sandbox does NOT defend against:
    - SOPHISTICATED: determined attackers with Python internals knowledge
    - KERNEL_EXPLOIT: attacks requiring kernel-level isolation
    """
    ACCIDENTAL = "accidental"
    OPPORTUNISTIC = "opportunistic"


# restricted builtins - remove dangerous functions
SAFE_BUILTINS = {
    "True": True,
    "False": False,
    "None": None,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "hash": hash,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "ord": ord,
    "pow": pow,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

# blocked patterns - these are checked before execution
BLOCKED_PATTERNS = frozenset([
    "import os",
    "import subprocess",
    "import sys",
    "import socket",
    "import ctypes",
    "import pickle",
    "import marshal",
    "__import__",
    "exec(",
    "eval(",
    "open(",
    "compile(",
    "getattr(",
    "setattr(",
    "delattr(",
    "__builtins__",
    "__globals__",
    "__code__",
    "__class__",
    "__subclasses__",
    "__mro__",
    "__bases__",
    "breakpoint(",
])


class SecurityViolation(Exception):
    """raised when sandbox detects blocked operation."""
    pass


class TimeoutError(Exception):
    """raised when execution exceeds timeout."""
    pass


@dataclass
class IsolationConfig:
    """configuration for soft isolation."""
    timeout_seconds: float = 30.0
    blocked_patterns: frozenset[str] = BLOCKED_PATTERNS
    use_firejail: bool = False
    firejail_seccomp: str = "execve,ptrace,socket"
    allowed_paths: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """result of isolated execution."""
    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    isolation_level: str = "python"
    locals: dict[str, Any] = field(default_factory=dict)


class SoftIsolation:
    """
    soft isolation sandbox with explicit threat model.

    THREAT MODEL:
    - Defends: accidental misuse, basic injection, obvious exploits
    - Does NOT defend: sophisticated attacks, pickle exploits, ctypes

    LIMITATIONS (documented, not hidden):
    - Python sandboxing is fundamentally limited by interpreter design
    - Determined attacker with code execution can likely escape
    - No kernel-level isolation without firejail
    - firejail provides stronger isolation but requires installation

    USE CASES:
    - Defense in depth (layer among other protections)
    - Audit logging of execution attempts
    - Blocking obvious dangerous patterns
    - Timeout enforcement for runaway code
    """

    def __init__(self, config: IsolationConfig | None = None):
        self._config = config or IsolationConfig()
        self._execution_count = 0

        # check firejail availability at init
        self._firejail_available = shutil.which("firejail") is not None

        if self._config.use_firejail and not self._firejail_available:
            raise RuntimeError(
                "firejail requested but not installed. "
                "Either install firejail or set use_firejail=False"
            )

        logger.info(
            f"soft isolation initialized: firejail={'available' if self._firejail_available else 'unavailable'}"
        )

    @property
    def threat_model(self) -> ThreatModel:
        """return the threat model this sandbox defends against."""
        return ThreatModel.OPPORTUNISTIC

    def get_security_disclaimer(self) -> str:
        """return honest security limitations."""
        return """
SECURITY LIMITATIONS (read before use):

This provides SOFT ISOLATION, not cryptographic security.

DEFENDED THREATS:
- Accidental dangerous operations
- Basic pattern-based injection attacks
- Runaway code (timeout enforced)
- Obvious import/exec/eval attempts

NOT DEFENDED:
- Pickle deserialization attacks
- ctypes/cffi escape vectors
- CPython interpreter internals exploitation
- Attribute access chains (__class__.__mro__ etc may have bypasses)
- Memory corruption or timing attacks

FOR HOSTILE CODE:
- Use firejail with seccomp (stronger but not perfect)
- Use VMs with snapshot/restore (strongest practical option)
- Never trust sandbox alone for security-critical isolation
"""

    def _check_blocked_patterns(self, code: str) -> None:
        """check code for blocked patterns, raise SecurityViolation if found."""
        code_lower = code.lower()
        for pattern in self._config.blocked_patterns:
            if pattern.lower() in code_lower:
                raise SecurityViolation(f"blocked pattern: {pattern}")

    def execute(self, code: str) -> ExecutionResult:
        """
        execute code in isolated environment.

        uses firejail if configured and available, else Python sandbox.
        """
        start_time = time.time()
        self._execution_count += 1

        # check blocked patterns first
        try:
            self._check_blocked_patterns(code)
        except SecurityViolation as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                isolation_level="blocked",
            )

        # choose isolation method
        if self._config.use_firejail and self._firejail_available:
            result = self._execute_firejail(code)
        else:
            result = self._execute_python(code)

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    def _execute_python(self, code: str) -> ExecutionResult:
        """execute in Python sandbox with timeout."""
        result_container: dict[str, Any] = {"done": False, "result": None, "error": None}
        restricted_globals = {"__builtins__": SAFE_BUILTINS}
        restricted_locals: dict[str, Any] = {}

        def run_code():
            try:
                compiled = compile(code, "<sandbox>", "exec")
                exec(compiled, restricted_globals, restricted_locals)
                result_container["result"] = restricted_locals.get("result")
                result_container["locals"] = {
                    k: v for k, v in restricted_locals.items()
                    if not k.startswith("_")
                }
            except Exception as e:
                result_container["error"] = str(e)
            finally:
                result_container["done"] = True

        thread = threading.Thread(target=run_code, daemon=True)
        thread.start()
        thread.join(timeout=self._config.timeout_seconds)

        if not result_container["done"]:
            return ExecutionResult(
                success=False,
                error=f"timeout after {self._config.timeout_seconds}s",
                isolation_level="python",
            )

        if result_container["error"]:
            return ExecutionResult(
                success=False,
                error=result_container["error"],
                isolation_level="python",
            )

        return ExecutionResult(
            success=True,
            result=result_container["result"],
            locals=result_container.get("locals", {}),
            isolation_level="python",
        )

    def _execute_firejail(self, code: str) -> ExecutionResult:
        """execute in firejail sandbox."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            cmd = [
                "firejail",
                "--quiet",
                "--net=none",
                f"--seccomp.drop={self._config.firejail_seccomp}",
                "--",
                "python3",
                script_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )

            if result.returncode == 0:
                return ExecutionResult(
                    success=True,
                    result=result.stdout.strip() if result.stdout else None,
                    isolation_level="firejail",
                )
            else:
                return ExecutionResult(
                    success=False,
                    error=result.stderr or f"exit code {result.returncode}",
                    isolation_level="firejail",
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"timeout after {self._config.timeout_seconds}s",
                isolation_level="firejail",
            )
        finally:
            os.unlink(script_path)

    def is_firejail_available(self) -> bool:
        """check if firejail is installed."""
        return self._firejail_available

    @property
    def execution_count(self) -> int:
        """return total executions."""
        return self._execution_count


class TrustBoundary:
    """
    defines and enforces trust boundaries in the system.

    trust zones:
    - TRUSTED: core logic, verified invariants, signed code
    - SEMI_TRUSTED: user config, validated inputs
    - UNTRUSTED: external inputs, user code, network data

    data crossing boundaries must be validated and sanitized.
    """

    def __init__(self):
        self._validators: dict[str, Callable[[Any], tuple[bool, str]]] = {}
        self._sanitizers: dict[str, Callable[[Any], Any]] = {}
        self._crossings: list[dict[str, Any]] = []

    def register_validator(
        self,
        boundary: str,
        validator: Callable[[Any], tuple[bool, str]],
    ) -> None:
        """register validator for boundary crossing."""
        self._validators[boundary] = validator

    def register_sanitizer(
        self,
        boundary: str,
        sanitizer: Callable[[Any], Any],
    ) -> None:
        """register sanitizer for boundary crossing."""
        self._sanitizers[boundary] = sanitizer

    def cross_boundary(
        self,
        boundary: str,
        data: Any,
        source: str,
        target: str,
    ) -> tuple[Any, bool, str]:
        """
        validate and sanitize data crossing a trust boundary.

        returns (sanitized_data, success, message)
        """
        crossing = {
            "boundary": boundary,
            "source": source,
            "target": target,
            "timestamp": time.time(),
        }

        # validate
        if boundary in self._validators:
            valid, msg = self._validators[boundary](data)
            if not valid:
                crossing["result"] = "rejected"
                crossing["reason"] = msg
                self._crossings.append(crossing)
                return None, False, msg

        # sanitize
        result = data
        if boundary in self._sanitizers:
            result = self._sanitizers[boundary](data)

        crossing["result"] = "accepted"
        self._crossings.append(crossing)

        return result, True, "ok"

    def get_crossing_log(self) -> list[dict[str, Any]]:
        """return audit log of boundary crossings."""
        return self._crossings.copy()


# backwards compatibility
Sandbox = SoftIsolation
