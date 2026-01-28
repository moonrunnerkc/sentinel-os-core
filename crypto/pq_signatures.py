# Author: Bradley R. Kinnard
# post-quantum signatures using Dilithium (via cryptography library fallback)

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from utils.helpers import get_logger

logger = get_logger(__name__)


# note: true Dilithium requires additional libraries (liboqs, pqcrypto)
# this implementation uses Ed25519 as fallback with Dilithium interface
# when liboqs-python becomes available, swap the backend


@dataclass
class PQKeyPair:
    """
    post-quantum key pair container.

    when liboqs is available: uses Dilithium3
    fallback: uses Ed25519 (not PQ-secure, but same interface)
    """
    public_key: bytes
    private_key: bytes
    algorithm: str
    created_at: float
    key_id: str

    def public_key_hex(self) -> str:
        return self.public_key.hex()

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_key": self.public_key_hex(),
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "key_id": self.key_id,
        }


def _try_dilithium() -> bool:
    """check if liboqs is available for true PQ crypto."""
    try:
        import oqs  # noqa: F401
        return True
    except ImportError:
        return False


def generate_keypair(key_id: str | None = None) -> PQKeyPair:
    """
    generate a post-quantum keypair.

    tries Dilithium3 via liboqs, falls back to Ed25519.
    """
    timestamp = time.time()
    kid = key_id or f"pq_{int(timestamp * 1000)}"

    if _try_dilithium():
        # use actual Dilithium
        import oqs
        signer = oqs.Signature("Dilithium3")
        public_key = signer.generate_keypair()
        private_key = signer.export_secret_key()

        logger.info(f"generated Dilithium3 keypair: {kid}")

        return PQKeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm="Dilithium3",
            created_at=timestamp,
            key_id=kid,
        )
    else:
        # fallback to Ed25519
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        priv_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        logger.warning(f"generated Ed25519 keypair (PQ fallback): {kid}")

        return PQKeyPair(
            public_key=pub_bytes,
            private_key=priv_bytes,
            algorithm="Ed25519-PQ-Fallback",
            created_at=timestamp,
            key_id=kid,
        )


class PQSigner:
    """
    post-quantum message signer.

    signs messages with Dilithium or Ed25519 fallback.
    """

    def __init__(self, keypair: PQKeyPair):
        self._keypair = keypair
        self._use_dilithium = keypair.algorithm == "Dilithium3"

        if self._use_dilithium:
            import oqs
            self._signer = oqs.Signature("Dilithium3", keypair.private_key)
        else:
            self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
                keypair.private_key
            )

    def sign(self, message: bytes) -> bytes:
        """sign a message."""
        if self._use_dilithium:
            signature = self._signer.sign(message)
        else:
            signature = self._private_key.sign(message)

        return signature

    def sign_json(self, data: dict[str, Any]) -> dict[str, Any]:
        """sign a JSON-serializable dict and return signed envelope."""
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        message = canonical.encode()
        signature = self.sign(message)

        return {
            "payload": data,
            "signature": signature.hex(),
            "algorithm": self._keypair.algorithm,
            "key_id": self._keypair.key_id,
            "signed_at": time.time(),
        }


class PQVerifier:
    """
    post-quantum signature verifier.

    verifies signatures without needing private key.
    """

    def __init__(self, public_key: bytes, algorithm: str):
        self._public_key = public_key
        self._algorithm = algorithm
        self._use_dilithium = algorithm == "Dilithium3"

        if self._use_dilithium:
            import oqs
            self._verifier = oqs.Signature("Dilithium3")
        else:
            self._ed_public = ed25519.Ed25519PublicKey.from_public_bytes(public_key)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """verify a signature."""
        try:
            if self._use_dilithium:
                return self._verifier.verify(message, signature, self._public_key)
            else:
                self._ed_public.verify(signature, message)
                return True
        except Exception as e:
            logger.warning(f"signature verification failed: {e}")
            return False

    def verify_json(self, signed_envelope: dict[str, Any]) -> tuple[bool, str]:
        """verify a signed JSON envelope."""
        try:
            payload = signed_envelope["payload"]
            signature = bytes.fromhex(signed_envelope["signature"])

            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            message = canonical.encode()

            if self.verify(message, signature):
                return True, "signature valid"
            else:
                return False, "signature invalid"

        except KeyError as e:
            return False, f"missing field: {e}"
        except Exception as e:
            return False, f"verification error: {e}"

    @classmethod
    def from_keypair(cls, keypair: PQKeyPair) -> "PQVerifier":
        """create verifier from keypair (uses public key only)."""
        return cls(keypair.public_key, keypair.algorithm)


class SignedLogChain:
    """
    chain of signed log entries with PQ signatures.

    each entry is signed and includes hash of previous entry,
    creating a tamper-evident chain.
    """

    def __init__(self, signer: PQSigner):
        self._signer = signer
        self._chain: list[dict[str, Any]] = []
        self._prev_hash: str = "genesis"

    def _compute_hash(self, data: dict[str, Any]) -> str:
        """compute hash of log entry."""
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        """append a signed entry to the chain."""
        entry_with_chain = {
            "index": len(self._chain),
            "prev_hash": self._prev_hash,
            "entry": entry,
            "timestamp": time.time(),
        }

        signed = self._signer.sign_json(entry_with_chain)
        self._prev_hash = self._compute_hash(signed)
        self._chain.append(signed)

        return signed

    def get_chain(self) -> list[dict[str, Any]]:
        """return the full chain."""
        return self._chain.copy()

    def verify_chain(self, verifier: PQVerifier) -> tuple[bool, str]:
        """verify the entire chain."""
        if not self._chain:
            return True, "empty chain"

        prev_hash = "genesis"

        for i, entry in enumerate(self._chain):
            # verify signature
            valid, msg = verifier.verify_json(entry)
            if not valid:
                return False, f"entry {i} signature invalid: {msg}"

            # verify chain linkage
            payload = entry["payload"]
            if payload["prev_hash"] != prev_hash:
                return False, f"chain break at entry {i}"

            prev_hash = self._compute_hash(entry)

        return True, f"chain valid ({len(self._chain)} entries)"

    def get_merkle_root(self) -> str:
        """compute merkle root of chain for compact verification."""
        from crypto.merkle import MerkleTree

        tree = MerkleTree()
        for entry in self._chain:
            tree.add_leaf(self._compute_hash(entry))
        tree.build()

        return tree.root
