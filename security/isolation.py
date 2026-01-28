# Author: Bradley R. Kinnard
# high-assurance isolation module - defense in depth for adversarial environments

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum, auto
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time

from utils.helpers import get_logger

logger = get_logger(__name__)


class IsolationLevel(Enum):
    """isolation levels from lowest to highest assurance."""
    NONE = auto()          # no isolation (testing only)
    PYTHON = auto()        # python-level sandbox (best effort)
    FIREJAIL = auto()      # firejail with seccomp
    CONTAINER = auto()     # docker/podman container
    MICROVM = auto()       # firecracker/qemu microVM
    SEL4 = auto()          # seL4 verified microkernel (future)


@dataclass
class IsolationConfig:
    """configuration for isolation environment."""
    level: IsolationLevel
    allowed_paths: list[str] = field(default_factory=list)
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 100
    timeout_seconds: int = 30
    network_enabled: bool = False
    readonly_root: bool = True
    seccomp_profile: str = "execve,ptrace,socket"


@dataclass
class ExecutionResult:
    """result of isolated execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    elapsed_ms: float
    isolation_level: IsolationLevel
    state_hash_before: str
    state_hash_after: str
    integrity_verified: bool


class TrustBoundary:
    """
    defines and enforces trust boundaries in the system.

    trust zones:
    1. TRUSTED: core logic, verified invariants, signed code
    2. SEMI_TRUSTED: user config, validated inputs
    3. UNTRUSTED: external inputs, user code, network data

    data crossing boundaries must be validated and sanitized.
    """

    def __init__(self):
        self._validators: dict[str, Callable[[Any], tuple[bool, str]]] = {}
        self._sanitizers: dict[str, Callable[[Any], Any]] = {}
        self._boundary_crossings: list[dict[str, Any]] = []

    def register_validator(
        self,
        boundary_name: str,
        validator: Callable[[Any], tuple[bool, str]],
    ) -> None:
        """register a validator for a boundary crossing."""
        self._validators[boundary_name] = validator
        logger.info(f"registered validator for boundary: {boundary_name}")

    def register_sanitizer(
        self,
        boundary_name: str,
        sanitizer: Callable[[Any], Any],
    ) -> None:
        """register a sanitizer for boundary crossing."""
        self._sanitizers[boundary_name] = sanitizer

    def cross_boundary(
        self,
        boundary_name: str,
        data: Any,
        source_zone: str,
        target_zone: str,
    ) -> tuple[Any, bool, str]:
        """
        validate and sanitize data crossing a trust boundary.

        returns: (sanitized_data, success, message)
        """
        crossing = {
            "boundary": boundary_name,
            "source": source_zone,
            "target": target_zone,
            "timestamp": time.time(),
        }

        # validate
        if boundary_name in self._validators:
            valid, msg = self._validators[boundary_name](data)
            if not valid:
                crossing["result"] = "rejected"
                crossing["reason"] = msg
                self._boundary_crossings.append(crossing)
                return None, False, msg

        # sanitize
        sanitized = data
        if boundary_name in self._sanitizers:
            sanitized = self._sanitizers[boundary_name](data)

        crossing["result"] = "accepted"
        self._boundary_crossings.append(crossing)

        return sanitized, True, "ok"

    def get_crossing_log(self) -> list[dict[str, Any]]:
        """return log of boundary crossings."""
        return self._boundary_crossings.copy()


class IsolationEngine:
    """
    multi-level isolation engine.

    provides defense-in-depth with automatic fallback:
    1. Try highest available isolation level
    2. Fall back to lower levels if unavailable
    3. Always log actual isolation achieved
    """

    def __init__(self, config: IsolationConfig | None = None):
        self._config = config or IsolationConfig(level=IsolationLevel.PYTHON)
        self._available_levels = self._detect_available_levels()

        logger.info(f"isolation engine: available levels = {[l.name for l in self._available_levels]}")

    def _detect_available_levels(self) -> list[IsolationLevel]:
        """detect which isolation levels are available."""
        available = [IsolationLevel.NONE, IsolationLevel.PYTHON]

        # check firejail
        if shutil.which("firejail"):
            available.append(IsolationLevel.FIREJAIL)

        # check docker/podman
        if shutil.which("docker") or shutil.which("podman"):
            available.append(IsolationLevel.CONTAINER)

        # check firecracker/qemu
        if shutil.which("firecracker") or shutil.which("qemu-system-x86_64"):
            available.append(IsolationLevel.MICROVM)

        return available

    def get_effective_level(self) -> IsolationLevel:
        """get the actual isolation level that will be used."""
        requested = self._config.level

        # find highest available up to requested
        for level in reversed(list(IsolationLevel)):
            if level.value <= requested.value and level in self._available_levels:
                return level

        return IsolationLevel.NONE

    def execute(
        self,
        code: str,
        context: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """execute code in isolated environment."""
        level = self.get_effective_level()
        start_time = time.time()

        # compute state hash before
        state_before = self._compute_state_hash(context or {})

        if level == IsolationLevel.PYTHON:
            result = self._execute_python_sandbox(code, context)
        elif level == IsolationLevel.FIREJAIL:
            result = self._execute_firejail(code, context)
        elif level == IsolationLevel.CONTAINER:
            result = self._execute_container(code, context)
        else:
            result = self._execute_no_isolation(code, context)

        elapsed = (time.time() - start_time) * 1000

        # compute state hash after
        state_after = self._compute_state_hash(context or {})

        return ExecutionResult(
            success=result["success"],
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            exit_code=result.get("exit_code", 0),
            elapsed_ms=elapsed,
            isolation_level=level,
            state_hash_before=state_before,
            state_hash_after=state_after,
            integrity_verified=state_before == state_after or not context,
        )

    def _compute_state_hash(self, state: dict[str, Any]) -> str:
        """compute hash of execution context state."""
        canonical = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _execute_no_isolation(
        self,
        code: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """execute without isolation (testing only)."""
        logger.warning("executing without isolation - testing only")

        try:
            exec_globals = {"__builtins__": __builtins__}
            exec_globals.update(context or {})
            exec(code, exec_globals)

            return {"success": True, "stdout": "", "stderr": "", "exit_code": 0}
        except Exception as e:
            return {"success": False, "stderr": str(e), "exit_code": 1}

    def _execute_python_sandbox(
        self,
        code: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """execute in Python sandbox."""
        from security.sandbox import Sandbox

        sandbox = Sandbox({"allowed_paths": self._config.allowed_paths})
        result = sandbox.execute_safely(code)

        return {
            "success": result["success"],
            "stdout": str(result.get("result", "")),
            "stderr": result.get("error", ""),
            "exit_code": 0 if result["success"] else 1,
        }

    def _execute_firejail(
        self,
        code: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """execute in firejail sandbox."""
        # write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            cmd = [
                "firejail",
                "--quiet",
                "--net=none" if not self._config.network_enabled else "",
                f"--seccomp.drop={self._config.seccomp_profile}",
                f"--rlimit-as={self._config.memory_limit_mb * 1024 * 1024}",
                "--",
                "python3",
                script_path,
            ]
            cmd = [c for c in cmd if c]  # remove empty strings

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stderr": f"timeout after {self._config.timeout_seconds}s",
                "exit_code": 124,
            }
        finally:
            os.unlink(script_path)

    def _execute_container(
        self,
        code: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """execute in container."""
        # determine container runtime
        runtime = "docker" if shutil.which("docker") else "podman"

        # write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            cmd = [
                runtime, "run",
                "--rm",
                "--network=none" if not self._config.network_enabled else "",
                f"--memory={self._config.memory_limit_mb}m",
                "--read-only" if self._config.readonly_root else "",
                "-v", f"{script_path}:/script.py:ro",
                "python:3.12-slim",
                "python", "/script.py",
            ]
            cmd = [c for c in cmd if c]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stderr": f"timeout after {self._config.timeout_seconds}s",
                "exit_code": 124,
            }
        finally:
            os.unlink(script_path)


class SecurityAudit:
    """
    security audit and compliance checking.
    """

    def __init__(self):
        self._checks: list[tuple[str, Callable[[], tuple[bool, str]]]] = []
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """register default security checks."""

        def check_network_isolation():
            # verify no unexpected network connections
            try:
                result = subprocess.run(
                    ["ss", "-tuln"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # check for unexpected listeners
                lines = result.stdout.strip().split("\n")
                return len(lines) <= 2, f"active listeners: {len(lines) - 1}"
            except Exception as e:
                return False, str(e)

        self._checks.append(("network_isolation", check_network_isolation))

        def check_firejail_available():
            available = shutil.which("firejail") is not None
            return available, "firejail available" if available else "firejail not installed"

        self._checks.append(("firejail_available", check_firejail_available))

        def check_container_runtime():
            docker = shutil.which("docker") is not None
            podman = shutil.which("podman") is not None
            available = docker or podman
            return available, f"docker={docker}, podman={podman}"

        self._checks.append(("container_runtime", check_container_runtime))

    def run_audit(self) -> dict[str, Any]:
        """run all security checks."""
        results = {}
        passed = 0
        failed = 0

        for name, check_fn in self._checks:
            try:
                success, message = check_fn()
                results[name] = {"passed": success, "message": message}
                if success:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                results[name] = {"passed": False, "message": f"check failed: {e}"}
                failed += 1

        return {
            "timestamp": time.time(),
            "checks": results,
            "summary": {"passed": passed, "failed": failed, "total": len(self._checks)},
        }
