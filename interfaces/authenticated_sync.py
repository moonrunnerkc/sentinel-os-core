# Author: Bradley R. Kinnard
# authenticated sync - signed belief export/import with replay protection
# replaces "ZKP-based" sync with honest signature-based integrity

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any
from collections import OrderedDict

from crypto.pq_signatures import Signer, Verifier, KeyPair, Algorithm
from utils.helpers import get_logger


logger = get_logger(__name__)


class SyncError(Exception):
    """raised when sync operations fail."""
    pass


class ReplayDetectedError(SyncError):
    """raised when a replayed message is detected."""
    pass


class SignatureInvalidError(SyncError):
    """raised when signature verification fails."""
    pass


@dataclass
class SignedExport:
    """
    signed belief export for cross-device sync.

    contains:
    - payload: the actual beliefs being exported
    - signature: Ed25519/Dilithium signature
    - nonce: unique per-export for replay prevention
    - timestamp: when export was created
    """
    payload: dict
    signature: str  # hex-encoded
    signer_key_id: str
    algorithm: str
    nonce: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "signature": self.signature,
            "signer_key_id": self.signer_key_id,
            "algorithm": self.algorithm,
            "nonce": self.nonce,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SignedExport":
        return cls(
            payload=d["payload"],
            signature=d["signature"],
            signer_key_id=d["signer_key_id"],
            algorithm=d["algorithm"],
            nonce=d["nonce"],
            timestamp=d["timestamp"],
        )


@dataclass
class SyncResult:
    """result of an import operation."""
    success: bool
    imported_count: int = 0
    rejected_count: int = 0
    rejection_reasons: list[str] = field(default_factory=list)


class NonceTracker:
    """
    tracks seen nonces for replay prevention.

    uses LRU eviction to bound memory usage.
    """

    def __init__(
        self,
        max_nonces: int = 10000,
        max_age_seconds: float = 86400,  # 24 hours
    ):
        self._max_nonces = max_nonces
        self._max_age = max_age_seconds
        self._seen: OrderedDict[str, float] = OrderedDict()

    def is_replay(self, nonce: str, timestamp: float) -> bool:
        """check if nonce is a replay."""
        # check age first
        now = time.time()
        if now - timestamp > self._max_age:
            return True  # too old

        # check if seen
        if nonce in self._seen:
            return True

        return False

    def mark_seen(self, nonce: str, timestamp: float) -> None:
        """mark nonce as seen."""
        self._seen[nonce] = timestamp

        # evict oldest if over capacity
        while len(self._seen) > self._max_nonces:
            self._seen.popitem(last=False)

    def prune_expired(self) -> int:
        """remove expired nonces. returns count pruned."""
        now = time.time()
        expired = [n for n, ts in self._seen.items() if now - ts > self._max_age]
        for n in expired:
            del self._seen[n]
        return len(expired)

    def to_dict(self) -> dict[str, float]:
        """export for persistence."""
        return dict(self._seen)

    def load_dict(self, d: dict[str, float]) -> None:
        """load from persisted state."""
        self._seen = OrderedDict(d)
        self.prune_expired()


class AuthenticatedSync:
    """
    authenticated belief synchronization using real signatures.

    features:
    - signs exports with Ed25519/Dilithium
    - verifies signatures on import
    - detects replayed messages via nonce tracking
    - all operations logged for audit

    NOT zero-knowledge: recipient sees full belief content.
    integrity and authenticity are guaranteed, not privacy.
    """

    def __init__(
        self,
        signer: Signer,
        own_keypair: KeyPair,
        seed: int = 42,
    ):
        self._signer = signer
        self._keypair = own_keypair
        self._seed = seed
        self._counter = 0
        self._nonce_tracker = NonceTracker()
        self._known_keys: dict[str, KeyPair] = {
            own_keypair.key_id: own_keypair,
        }

        import numpy as np
        self._rng = np.random.RandomState(seed)

    def register_peer_key(self, keypair: KeyPair) -> None:
        """register a peer's public key for verification."""
        self._known_keys[keypair.key_id] = keypair
        logger.info(f"registered peer key: {keypair.key_id}")

    def _generate_nonce(self) -> str:
        """generate unique nonce (deterministic under seed for testing)."""
        self._counter += 1
        random_part = self._rng.bytes(8).hex()
        return f"{int(time.time() * 1000)}_{self._counter}_{random_part}"

    def _canonical(self, data: dict) -> str:
        """produce canonical JSON for signing."""
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def export_beliefs(
        self,
        beliefs: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> SignedExport:
        """
        export beliefs as a signed package.

        beliefs: list of belief dicts to export
        metadata: optional additional metadata
        """
        timestamp = time.time()
        nonce = self._generate_nonce()

        payload = {
            "beliefs": beliefs,
            "metadata": metadata or {},
            "export_timestamp": timestamp,
            "nonce": nonce,
        }

        # sign the canonical payload
        canonical = self._canonical(payload)
        signature = self._signer.sign(canonical.encode())

        export = SignedExport(
            payload=payload,
            signature=signature.hex(),
            signer_key_id=self._signer.key_id,
            algorithm=self._signer.algorithm.value,
            nonce=nonce,
            timestamp=timestamp,
        )

        logger.info(f"exported {len(beliefs)} beliefs, nonce={nonce[:16]}...")
        return export

    def import_beliefs(
        self,
        signed_export: SignedExport,
    ) -> tuple[SyncResult, list[dict[str, Any]]]:
        """
        import beliefs from a signed export.

        verifies signature and checks for replay.
        returns (result, imported_beliefs).
        """
        reasons: list[str] = []

        # check replay
        if self._nonce_tracker.is_replay(signed_export.nonce, signed_export.timestamp):
            raise ReplayDetectedError(
                f"replay detected: nonce={signed_export.nonce[:16]}..."
            )

        # lookup signer key
        if signed_export.signer_key_id not in self._known_keys:
            raise SignatureInvalidError(
                f"unknown signer: {signed_export.signer_key_id}"
            )

        peer_key = self._known_keys[signed_export.signer_key_id]

        # verify algorithm matches
        if signed_export.algorithm != peer_key.algorithm.value:
            raise SignatureInvalidError(
                f"algorithm mismatch: expected {peer_key.algorithm.value}, "
                f"got {signed_export.algorithm}"
            )

        # verify signature
        verifier = Verifier(peer_key.public_key, peer_key.algorithm)
        canonical = self._canonical(signed_export.payload)
        signature = bytes.fromhex(signed_export.signature)

        if not verifier.verify(canonical.encode(), signature):
            raise SignatureInvalidError("signature verification failed")

        # mark nonce as seen
        self._nonce_tracker.mark_seen(signed_export.nonce, signed_export.timestamp)

        # extract beliefs
        beliefs = signed_export.payload.get("beliefs", [])

        logger.info(
            f"imported {len(beliefs)} beliefs from {signed_export.signer_key_id}"
        )

        return SyncResult(
            success=True,
            imported_count=len(beliefs),
            rejected_count=0,
            rejection_reasons=reasons,
        ), beliefs

    def get_nonce_tracker(self) -> NonceTracker:
        """access nonce tracker for persistence."""
        return self._nonce_tracker


# backwards compatibility alias
FederatedSync = AuthenticatedSync
