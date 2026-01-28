# Author: Bradley R. Kinnard
# sandbox - restricted code execution with best-effort isolation

import hashlib
import shutil
import subprocess
from typing import Any

from utils.helpers import get_logger


logger = get_logger(__name__)


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

# blocked patterns in code
BLOCKED_PATTERNS = [
    "import os",
    "import subprocess",
    "import sys",
    "import socket",
    "import ctypes",
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
]


class Sandbox:
    """
    restricted code execution sandbox.
    provides best-effort isolation, not cryptographic security.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._allowed_paths: list[str] = self._config.get("allowed_paths", [])
        self._use_firejail = self._config.get("use_firejail", False)
        self._state_hash = self._compute_state_hash()

    def _compute_state_hash(self) -> str:
        """compute hash of relevant global state."""
        # simplified state hash - track sandbox config
        state_repr = str(sorted(self._config.items()))
        return hashlib.sha256(state_repr.encode()).hexdigest()

    def get_state_hash(self) -> str:
        """return current state hash."""
        return self._compute_state_hash()

    def add_allowed_path(self, path: str) -> None:
        """add a path to the allowed list."""
        self._allowed_paths.append(path)

    def execute_safely(self, code: str) -> dict[str, Any]:
        """
        execute code in a restricted environment.
        returns dict with success status and result/error.
        """
        # check for blocked patterns
        code_lower = code.lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                logger.warning(f"blocked pattern detected: {pattern}")
                return {
                    "success": False,
                    "error": f"blocked: {pattern} not allowed in sandbox"
                }

        # prepare restricted globals
        restricted_globals = {"__builtins__": SAFE_BUILTINS}
        restricted_locals: dict[str, Any] = {}

        try:
            # compile and execute
            compiled = compile(code, "<sandbox>", "exec")
            exec(compiled, restricted_globals, restricted_locals)

            # extract result if present
            result = restricted_locals.get("result", None)

            logger.debug(f"executed code successfully: {code[:50]}...")
            return {
                "success": True,
                "result": result,
                "locals": restricted_locals
            }

        except Exception as e:
            logger.warning(f"execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def is_firejail_available(self) -> bool:
        """check if firejail is installed."""
        return shutil.which("firejail") is not None

    def execute_with_firejail(
        self,
        code: str,
        timeout: int = 10
    ) -> dict[str, Any]:
        """execute code wrapped in firejail for enhanced isolation."""
        if not self.is_firejail_available():
            return {
                "success": False,
                "error": "firejail not installed"
            }

        seccomp_profile = self._config.get("seccomp_profile", "execve,ptrace")

        try:
            result = subprocess.run(
                [
                    "firejail",
                    "--net=none",
                    f"--seccomp.drop={seccomp_profile}",
                    "--quiet",
                    "--",
                    "python3",
                    "-c",
                    code
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or "execution failed",
                    "returncode": result.returncode
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"execution timed out after {timeout}s"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_security_disclaimer(self) -> str:
        """return security limitations disclaimer."""
        return """
SECURITY DISCLAIMER:
This sandbox provides best-effort isolation, NOT cryptographic security.

Known limitations:
- Python-based sandboxing cannot prevent all CPython escape vectors
- Determined adversaries may bypass restrictions via interpreter internals
- Pickle deserialization can execute arbitrary code
- No kernel-level isolation without firejail

For truly hostile code execution, consider:
- Dedicated VMs with snapshot/restore
- WebAssembly or gVisor sandboxes
- Hardware-enforced isolation (SGX enclaves)

This sandbox is suitable for:
- Defense-in-depth against casual misuse
- Logging and auditing code execution attempts
- Blocking obvious dangerous patterns
"""
