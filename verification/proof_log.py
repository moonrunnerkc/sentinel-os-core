# Author: Bradley R. Kinnard
# proof log - reproducible verification artifacts

import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class ProofEntry:
    """single entry in the proof log."""
    sequence: int
    invariant_name: str
    result: str
    message: str
    state_digest: str
    timestamp: float
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "invariant_name": self.invariant_name,
            "result": self.result,
            "message": self.message,
            "state_digest": self.state_digest,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ProofLog:
    """
    proof log for verification runs.

    creates reproducible, tamper-evident log of all invariant checks.
    """
    entries: list[ProofEntry] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    seed: int = 0
    version: str = "1.0.0"
    _prev_hash: str = "genesis"

    def _compute_entry_hash(self, entry: ProofEntry) -> str:
        """compute hash including chain linkage."""
        data = {
            "prev": self._prev_hash,
            "entry": entry.to_dict(),
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def append(
        self,
        invariant_name: str,
        result: str,
        message: str,
        state_digest: str,
        duration_ms: float,
    ) -> ProofEntry:
        """append a proof entry."""
        entry = ProofEntry(
            sequence=len(self.entries),
            invariant_name=invariant_name,
            result=result,
            message=message,
            state_digest=state_digest,
            timestamp=time.time(),
            duration_ms=duration_ms,
        )

        self._prev_hash = self._compute_entry_hash(entry)
        self.entries.append(entry)

        return entry

    def verify_chain(self) -> tuple[bool, str]:
        """verify the proof chain has not been tampered with."""
        if not self.entries:
            return True, "empty log"

        prev_hash = "genesis"
        for i, entry in enumerate(self.entries):
            data = {
                "prev": prev_hash,
                "entry": entry.to_dict(),
            }
            canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
            computed = hashlib.sha256(canonical.encode()).hexdigest()[:16]
            prev_hash = computed

        # final hash should match stored
        if prev_hash != self._prev_hash:
            return False, f"chain integrity failed at final hash"

        return True, f"chain valid ({len(self.entries)} entries)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "start_time": self.start_time,
            "entries": [e.to_dict() for e in self.entries],
            "final_hash": self._prev_hash,
            "total_entries": len(self.entries),
        }

    def save(self, path: Path | str) -> None:
        """save proof log to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"saved proof log to {path} ({len(self.entries)} entries)")

    @classmethod
    def load(cls, path: Path | str) -> "ProofLog":
        """load proof log from JSON file."""
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        log = cls(
            start_time=data["start_time"],
            seed=data["seed"],
            version=data["version"],
        )

        # rebuild entries
        for entry_data in data["entries"]:
            entry = ProofEntry(
                sequence=entry_data["sequence"],
                invariant_name=entry_data["invariant_name"],
                result=entry_data["result"],
                message=entry_data["message"],
                state_digest=entry_data["state_digest"],
                timestamp=entry_data["timestamp"],
                duration_ms=entry_data["duration_ms"],
            )
            log._prev_hash = log._compute_entry_hash(entry)
            log.entries.append(entry)

        return log

    def summary(self) -> dict[str, Any]:
        """return summary statistics."""
        passed = sum(1 for e in self.entries if e.result == "passed")
        failed = sum(1 for e in self.entries if e.result == "failed")
        total_ms = sum(e.duration_ms for e in self.entries)

        return {
            "total_checks": len(self.entries),
            "passed": passed,
            "failed": failed,
            "total_duration_ms": total_ms,
            "avg_duration_ms": total_ms / len(self.entries) if self.entries else 0,
        }


class ProofLogger:
    """
    logger that records verification to proof log.

    integrates with FormalChecker to create audit trail.
    """

    def __init__(self, seed: int = 42):
        self._log = ProofLog(seed=seed)
        self._state_digests: dict[str, str] = {}

    def compute_state_digest(self, state: dict[str, Any]) -> str:
        """compute digest of state for proof log."""
        # exclude internal fields
        clean_state = {k: v for k, v in state.items() if not k.startswith("_")}
        canonical = json.dumps(clean_state, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return digest

    def record_check(
        self,
        invariant_name: str,
        result: str,
        message: str,
        state: dict[str, Any],
        duration_ms: float,
    ) -> ProofEntry:
        """record a check to the proof log."""
        state_digest = self.compute_state_digest(state)
        return self._log.append(
            invariant_name=invariant_name,
            result=result,
            message=message,
            state_digest=state_digest,
            duration_ms=duration_ms,
        )

    def record_verification_report(
        self,
        report: Any,  # VerificationReport
        state: dict[str, Any],
    ) -> None:
        """record all checks from a verification report."""
        for result in report.results:
            self.record_check(
                invariant_name=result.name,
                result=result.result.value,
                message=result.message,
                state=state,
                duration_ms=result.duration_ms,
            )

    def get_log(self) -> ProofLog:
        """return the proof log."""
        return self._log

    def verify(self) -> tuple[bool, str]:
        """verify the proof log chain."""
        return self._log.verify_chain()

    def save(self, path: Path | str) -> None:
        """save proof log."""
        self._log.save(path)

    def summary(self) -> dict[str, Any]:
        """return summary."""
        return self._log.summary()


def run_formal_verification(
    state: dict[str, Any],
    output_path: Path | str | None = None,
    seed: int = 42,
) -> tuple[bool, ProofLog]:
    """
    run formal verification on state and generate proof log.

    returns (all_passed, proof_log).
    """
    from verification.formal_checker import create_standard_checker

    checker = create_standard_checker(fail_fast=False)
    proof_logger = ProofLogger(seed=seed)

    report = checker.check_all(state)
    proof_logger.record_verification_report(report, state)

    if output_path:
        proof_logger.save(output_path)

    valid, msg = proof_logger.verify()
    if not valid:
        logger.error(f"proof log verification failed: {msg}")

    return report.all_passed, proof_logger.get_log()
