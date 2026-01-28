# Author: Bradley R. Kinnard
# audit logger - tamper-evident logging with HMAC

import hmac
import hashlib
import json
import uuid
import time
import logging
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from utils.helpers import get_logger


logger = get_logger(__name__)


class AuditLogger:
    """
    tamper-evident audit logger with HMAC signing.
    supports session-based key rotation.
    """

    def __init__(
        self,
        master_seed: int = 42,
        pq_crypto: bool = False,
        log_dir: Path | str = "data/logs"
    ):
        self._master_seed = master_seed
        self._pq_crypto = pq_crypto
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._session_id = str(uuid.uuid4())
        self._session_key = self._derive_session_key()
        self._entries: list[dict[str, Any]] = []

        logger.info(f"audit logger initialized, session={self._session_id[:8]}...")
        if not pq_crypto:
            logger.info("using SHA256 fallback (PQ unavailable)")

    def _derive_session_key(self) -> bytes:
        """derive session-specific key from master seed."""
        if self._pq_crypto:
            # placeholder for kyber/dilithium when available
            # fall back to HKDF for now
            logger.debug("PQ crypto requested but using HKDF fallback")

        kdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=self._session_id.encode()
        )
        return kdf.derive(self._master_seed.to_bytes(32, "big"))

    def _compute_hmac(self, data: str) -> str:
        """compute HMAC for data using session key."""
        return hmac.new(
            self._session_key,
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    def log(
        self,
        event: str,
        level: str = "INFO",
        details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """log an event with HMAC signature."""
        entry = {
            "timestamp": time.time(),
            "session_id": self._session_id,
            "event": event,
            "level": level,
            "details": details or {},
            "sequence": len(self._entries)
        }

        # compute HMAC over serialized entry (without hmac field)
        entry_json = json.dumps(entry, sort_keys=True)
        entry["hmac"] = self._compute_hmac(entry_json)

        self._entries.append(entry)
        logger.log(
            getattr(logging, level, logging.INFO),
            f"audit: {event}"
        )

        return entry

    def verify_entry(self, entry: dict[str, Any]) -> bool:
        """verify HMAC of a log entry."""
        stored_hmac = entry.pop("hmac", None)
        if stored_hmac is None:
            return False

        entry_json = json.dumps(entry, sort_keys=True)
        expected_hmac = self._compute_hmac(entry_json)

        # restore hmac field
        entry["hmac"] = stored_hmac

        return hmac.compare_digest(stored_hmac, expected_hmac)

    def verify_all(self) -> tuple[bool, list[int]]:
        """verify all entries, return success and list of failed indices."""
        failed = []
        for i, entry in enumerate(self._entries):
            entry_copy = entry.copy()
            if not self.verify_entry(entry_copy):
                failed.append(i)

        return len(failed) == 0, failed

    def get_entries(self) -> list[dict[str, Any]]:
        """return all log entries."""
        return [e.copy() for e in self._entries]

    def save_to_disk(self, filename: str | None = None) -> Path:
        """save log entries to disk."""
        if filename is None:
            filename = f"audit_{self._session_id[:8]}_{int(time.time())}.json"

        path = self._log_dir / filename
        with open(path, "w") as f:
            json.dump({
                "session_id": self._session_id,
                "entries": self._entries,
                "pq_crypto": self._pq_crypto
            }, f, indent=2)

        logger.info(f"saved {len(self._entries)} audit entries to {path}")
        return path

    def load_from_disk(self, path: Path | str) -> None:
        """load log entries from disk."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        self._session_id = data["session_id"]
        self._entries = data["entries"]
        self._pq_crypto = data.get("pq_crypto", False)

        # re-derive session key for verification
        self._session_key = self._derive_session_key()

        logger.info(f"loaded {len(self._entries)} audit entries from {path}")

    def rotate_session(self) -> str:
        """rotate to a new session, generating new key."""
        old_session = self._session_id
        self._session_id = str(uuid.uuid4())
        self._session_key = self._derive_session_key()

        logger.info(f"rotated session: {old_session[:8]}... -> {self._session_id[:8]}...")
        return self._session_id

    def get_session_id(self) -> str:
        """return current session id."""
        return self._session_id
