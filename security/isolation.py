# Author: Bradley R. Kinnard
# isolation module - explicit levels, hard-fail on unavailable, no false claims

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from utils.helpers import get_logger

logger = get_logger(__name__)


class IsolationUnavailableError(RuntimeError):
    """raised when requested isolation level is unavailable."""
    pass


class IsolationLevel(Enum):
    """
    explicit isolation levels with documented threat models.

    each level has:
    - what it defends against (threat_model)
    - what it does NOT defend against (limitations)
    - external requirements (requires)

    levels never silently downgrade. if a level is unavailable
    and requested, IsolationUnavailableError is raised.
    """
    NONE = "none"
    PATTERN_ONLY = "pattern_only"
    PYTHON_SANDBOX = "python_sandbox"
    FIREJAIL = "firejail"
    DOCKER = "docker"


# threat models per level
LEVEL_THREAT_MODELS: dict[IsolationLevel, dict[str, Any]] = {
    IsolationLevel.NONE: {
        "defends": [],
        "limitations": [
            "no isolation at all",
            "use only for testing",
        ],
        "requires": [],
    },
    IsolationLevel.PATTERN_ONLY: {
        "defends": [
            "accidental dangerous imports",
            "obvious eval/exec patterns",
        ],
        "limitations": [
            "trivially bypassable by obfuscation",
            "no runtime restriction",
            "no sandbox",
        ],
        "requires": [],
    },
    IsolationLevel.PYTHON_SANDBOX: {
        "defends": [
            "accidental dangerous operations",
            "basic injection patterns",
            "obvious import attempts",
            "timeout enforcement",
        ],
        "limitations": [
            "python sandboxing is fundamentally limited",
            "determined attacker can escape via interpreter internals",
            "pickle/marshal deserialization attacks",
            "ctypes/cffi escape vectors",
            "no filesystem isolation beyond pattern blocking",
        ],
        "requires": [],
    },
    IsolationLevel.FIREJAIL: {
        "defends": [
            "filesystem access outside allowed paths",
            "network access (disabled)",
            "syscall filtering via seccomp",
            "namespace isolation",
            "all python_sandbox defenses",
        ],
        "limitations": [
            "kernel vulnerabilities can escape namespaces",
            "seccomp bypasses are occasionally discovered",
            "not a security boundary against kernel exploits",
        ],
        "requires": ["firejail binary"],
    },
    IsolationLevel.DOCKER: {
        "defends": [
            "process isolation via cgroups",
            "filesystem isolation via overlay",
            "network isolation (disabled)",
            "resource limits",
            "all firejail defenses",
        ],
        "limitations": [
            "container escapes exist (CVEs)",
            "docker daemon must be trusted",
            "not equivalent to VM isolation",
            "shared kernel with host",
        ],
        "requires": ["docker daemon"],
    },
}


def get_level_threat_model(level: IsolationLevel) -> dict[str, Any]:
    """return threat model for isolation level."""
    return LEVEL_THREAT_MODELS[level]


def check_level_available(level: IsolationLevel) -> tuple[bool, str]:
    """
    check if isolation level is available on this system.

    returns (available, reason).
    """
    if level == IsolationLevel.NONE:
        return True, "always available"

    if level == IsolationLevel.PATTERN_ONLY:
        return True, "always available"

    if level == IsolationLevel.PYTHON_SANDBOX:
        return True, "always available"

    if level == IsolationLevel.FIREJAIL:
        if shutil.which("firejail"):
            return True, "firejail binary found"
        return False, "firejail binary not found in PATH"

    if level == IsolationLevel.DOCKER:
        # check if docker daemon is running
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True, "docker daemon available"
            return False, "docker daemon not running"
        except FileNotFoundError:
            return False, "docker binary not found in PATH"
        except subprocess.TimeoutExpired:
            return False, "docker daemon not responding"
        except Exception as e:
            return False, f"docker check failed: {e}"

    return False, f"unknown level: {level}"


# restricted builtins for python sandbox
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

# blocked patterns
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


@dataclass
class IsolationConfig:
    """configuration for isolation."""
    level: IsolationLevel = IsolationLevel.PYTHON_SANDBOX
    timeout_seconds: float = 30.0
    blocked_patterns: frozenset[str] = BLOCKED_PATTERNS
    firejail_seccomp: str = "execve,ptrace,socket"
    docker_image: str = "python:3.12-slim"
    allowed_paths: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """result of isolated execution."""
    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    isolation_level: IsolationLevel = IsolationLevel.NONE
    locals: dict[str, Any] = field(default_factory=dict)


class Isolation:
    """
    isolation with explicit levels and hard-fail semantics.

    CRITICAL: this class never silently downgrades isolation levels.
    if a requested level is unavailable, IsolationUnavailableError is raised.

    THREAT MODEL:
    - each level has documented defenses and limitations
    - use get_level_threat_model() to understand what each level provides
    - no level provides security boundary against kernel exploits

    USAGE:
    1. choose level based on threat model
    2. check availability with check_level_available() if needed
    3. instantiate with IsolationConfig
    4. execute code
    """

    def __init__(self, config: IsolationConfig | None = None):
        self._config = config or IsolationConfig()
        self._execution_count = 0

        # hard-fail if requested level unavailable
        available, reason = check_level_available(self._config.level)
        if not available:
            raise IsolationUnavailableError(
                f"isolation level {self._config.level.value} unavailable: {reason}. "
                f"either install requirements or choose a different level."
            )

        self._level = self._config.level
        logger.info(f"isolation initialized: level={self._level.value}")

    @property
    def level(self) -> IsolationLevel:
        """return configured isolation level."""
        return self._level

    @property
    def threat_model(self) -> dict[str, Any]:
        """return threat model for configured level."""
        return get_level_threat_model(self._level)

    def get_security_disclaimer(self) -> str:
        """return honest security limitations for configured level."""
        model = self.threat_model
        defends = "\n".join(f"  - {d}" for d in model["defends"]) or "  (none)"
        limitations = "\n".join(f"  - {l}" for l in model["limitations"]) or "  (none)"

        return f"""
ISOLATION LEVEL: {self._level.value}

DEFENDED THREATS:
{defends}

LIMITATIONS (not defended):
{limitations}

GENERAL WARNING:
No Python-based isolation provides a security boundary.
For hostile code, use VMs with snapshot/restore.
"""

    def _check_blocked_patterns(self, code: str) -> None:
        """check code for blocked patterns."""
        if self._level == IsolationLevel.NONE:
            return  # no pattern checking in NONE mode

        code_lower = code.lower()
        for pattern in self._config.blocked_patterns:
            if pattern.lower() in code_lower:
                raise SecurityViolation(f"blocked pattern: {pattern}")

    def execute(self, code: str) -> ExecutionResult:
        """execute code at configured isolation level."""
        start_time = time.time()
        self._execution_count += 1

        # pattern check (unless NONE level)
        try:
            self._check_blocked_patterns(code)
        except SecurityViolation as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
                isolation_level=self._level,
            )

        # dispatch to level-specific executor
        if self._level == IsolationLevel.NONE:
            result = self._execute_none(code)
        elif self._level == IsolationLevel.PATTERN_ONLY:
            result = self._execute_pattern_only(code)
        elif self._level == IsolationLevel.PYTHON_SANDBOX:
            result = self._execute_python_sandbox(code)
        elif self._level == IsolationLevel.FIREJAIL:
            result = self._execute_firejail(code)
        elif self._level == IsolationLevel.DOCKER:
            result = self._execute_docker(code)
        else:
            result = ExecutionResult(
                success=False,
                error=f"unknown level: {self._level}",
                isolation_level=self._level,
            )

        result.execution_time_ms = (time.time() - start_time) * 1000
        result.isolation_level = self._level
        return result

    def _execute_none(self, code: str) -> ExecutionResult:
        """execute with no isolation (testing only)."""
        try:
            local_vars: dict[str, Any] = {}
            exec(code, {}, local_vars)
            return ExecutionResult(
                success=True,
                result=local_vars.get("result"),
                locals={k: v for k, v in local_vars.items() if not k.startswith("_")},
            )
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))

    def _execute_pattern_only(self, code: str) -> ExecutionResult:
        """execute with pattern blocking only, no sandbox."""
        # patterns already checked, execute normally
        return self._execute_none(code)

    def _execute_python_sandbox(self, code: str) -> ExecutionResult:
        """execute in python sandbox with restricted builtins."""
        result_container: dict[str, Any] = {
            "done": False,
            "result": None,
            "error": None,
            "locals": {},
        }
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
            )

        if result_container["error"]:
            return ExecutionResult(
                success=False,
                error=result_container["error"],
            )

        return ExecutionResult(
            success=True,
            result=result_container["result"],
            locals=result_container.get("locals", {}),
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
                )
            else:
                return ExecutionResult(
                    success=False,
                    error=result.stderr or f"exit code {result.returncode}",
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"timeout after {self._config.timeout_seconds}s",
            )
        finally:
            os.unlink(script_path)

    def _execute_docker(self, code: str) -> ExecutionResult:
        """execute in docker container."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            cmd = [
                "docker", "run",
                "--rm",
                "--network=none",
                "--memory=256m",
                "--cpus=1",
                "-v", f"{script_path}:/code.py:ro",
                self._config.docker_image,
                "python3", "/code.py",
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
                )
            else:
                return ExecutionResult(
                    success=False,
                    error=result.stderr or f"exit code {result.returncode}",
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"timeout after {self._config.timeout_seconds}s",
            )
        finally:
            os.unlink(script_path)

    def is_firejail_available(self) -> bool:
        """check if firejail is available."""
        available, _ = check_level_available(IsolationLevel.FIREJAIL)
        return available

    def is_docker_available(self) -> bool:
        """check if docker is available."""
        available, _ = check_level_available(IsolationLevel.DOCKER)
        return available

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


# backwards compatibility aliases
SoftIsolation = Isolation
Sandbox = Isolation
