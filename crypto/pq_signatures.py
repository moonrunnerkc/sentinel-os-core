# Author: Bradley R. Kinnard
# signature engine - explicit algorithm selection, no silent fallbacks

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from utils.helpers import get_logger

logger = get_logger(__name__)


class Algorithm(Enum):
    """supported signature algorithms."""
    ED25519 = "ed25519"
    DILITHIUM3 = "dilithium3"
    HYBRID_ED25519_DILITHIUM3 = "hybrid_ed25519_dilithium3"


class PQUnavailableError(ImportError):
    """
    raised when post-quantum crypto is requested but liboqs is not installed.

    this is NOT a silent fallback - the caller must handle this explicitly.
    """
    pass


def _liboqs_available() -> bool:
    """check if liboqs is installed for post-quantum crypto."""
    try:
        import oqs  # noqa: F401
        return True
    except ImportError:
        return False


def _require_liboqs(algorithm: Algorithm) -> None:
    """raise PQUnavailableError if algorithm requires liboqs and it's missing."""
    if algorithm in (Algorithm.DILITHIUM3, Algorithm.HYBRID_ED25519_DILITHIUM3):
        if not _liboqs_available():
            raise PQUnavailableError(
                f"{algorithm.value} requires liboqs-python. "
                "Install with: pip install liboqs-python (requires liboqs C library). "
                "No silent fallback to Ed25519 is permitted."
            )


@dataclass
class KeyPair:
    """
    signature key pair.

    algorithm is explicit - no silent fallback between algorithms.

    for hybrid mode, both ed25519 and dilithium3 keys are stored.
    """
    public_key: bytes
    private_key: bytes
    algorithm: Algorithm
    created_at: float
    key_id: str
    # hybrid mode keys (only populated for HYBRID_ED25519_DILITHIUM3)
    ed25519_public: bytes | None = None
    ed25519_private: bytes | None = None
    dilithium_public: bytes | None = None
    dilithium_private: bytes | None = None

    def public_key_hex(self) -> str:
        return self.public_key.hex()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "public_key": self.public_key_hex(),
            "algorithm": self.algorithm.value,
            "created_at": self.created_at,
            "key_id": self.key_id,
        }
        if self.algorithm == Algorithm.HYBRID_ED25519_DILITHIUM3:
            d["ed25519_public"] = self.ed25519_public.hex() if self.ed25519_public else None
            d["dilithium_public"] = self.dilithium_public.hex() if self.dilithium_public else None
        return d


def generate_keypair(
    algorithm: Algorithm = Algorithm.ED25519,
    key_id: str | None = None,
) -> KeyPair:
    """
    generate a key pair with explicit algorithm selection.

    raises PQUnavailableError if requested algorithm requires unavailable library.
    never silently falls back to a different algorithm.
    """
    _require_liboqs(algorithm)

    timestamp = time.time()
    kid = key_id or f"key_{int(timestamp * 1000)}"

    if algorithm == Algorithm.HYBRID_ED25519_DILITHIUM3:
        # generate both key types
        import oqs

        # ed25519
        ed_private = ed25519.Ed25519PrivateKey.generate()
        ed_public = ed_private.public_key()
        ed_priv_bytes = ed_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        ed_pub_bytes = ed_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        # dilithium3
        dil_signer = oqs.Signature("Dilithium3")
        dil_pub_bytes = dil_signer.generate_keypair()
        dil_priv_bytes = dil_signer.export_secret_key()

        # combined public key is concatenation (ed25519 || dilithium)
        combined_public = ed_pub_bytes + dil_pub_bytes
        combined_private = ed_priv_bytes + dil_priv_bytes

        logger.info(f"generated hybrid Ed25519+Dilithium3 keypair: {kid}")

        return KeyPair(
            public_key=combined_public,
            private_key=combined_private,
            algorithm=Algorithm.HYBRID_ED25519_DILITHIUM3,
            created_at=timestamp,
            key_id=kid,
            ed25519_public=ed_pub_bytes,
            ed25519_private=ed_priv_bytes,
            dilithium_public=dil_pub_bytes,
            dilithium_private=dil_priv_bytes,
        )

    elif algorithm == Algorithm.DILITHIUM3:
        import oqs
        signer = oqs.Signature("Dilithium3")
        public_key = signer.generate_keypair()
        private_key = signer.export_secret_key()

        logger.info(f"generated Dilithium3 keypair: {kid}")

        return KeyPair(
            public_key=public_key,
            private_key=private_key,
            algorithm=Algorithm.DILITHIUM3,
            created_at=timestamp,
            key_id=kid,
        )

    elif algorithm == Algorithm.ED25519:
        private_key_obj = ed25519.Ed25519PrivateKey.generate()
        public_key_obj = private_key_obj.public_key()

        priv_bytes = private_key_obj.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_bytes = public_key_obj.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        logger.info(f"generated Ed25519 keypair: {kid}")

        return KeyPair(
            public_key=pub_bytes,
            private_key=priv_bytes,
            algorithm=Algorithm.ED25519,
            created_at=timestamp,
            key_id=kid,
        )

    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")


class Signer:
    """
    message signer with explicit algorithm.

    for hybrid mode, signs with BOTH Ed25519 and Dilithium3.
    """

    def __init__(self, keypair: KeyPair):
        self._keypair = keypair
        self._algorithm = keypair.algorithm

        _require_liboqs(self._algorithm)

        if self._algorithm == Algorithm.HYBRID_ED25519_DILITHIUM3:
            import oqs
            self._ed_private = ed25519.Ed25519PrivateKey.from_private_bytes(
                keypair.ed25519_private
            )
            self._dil_signer = oqs.Signature("Dilithium3", keypair.dilithium_private)
        elif self._algorithm == Algorithm.DILITHIUM3:
            import oqs
            self._signer = oqs.Signature("Dilithium3", keypair.private_key)
        elif self._algorithm == Algorithm.ED25519:
            self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
                keypair.private_key
            )
        else:
            raise ValueError(f"unsupported algorithm: {self._algorithm}")

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    @property
    def key_id(self) -> str:
        return self._keypair.key_id

    def sign(self, message: bytes) -> bytes:
        """
        sign a message.

        for hybrid mode, returns concatenated signatures (ed25519 || dilithium).
        """
        if self._algorithm == Algorithm.HYBRID_ED25519_DILITHIUM3:
            ed_sig = self._ed_private.sign(message)
            dil_sig = self._dil_signer.sign(message)
            # length-prefixed concatenation for unambiguous parsing
            ed_len = len(ed_sig).to_bytes(4, 'big')
            return ed_len + ed_sig + dil_sig
        elif self._algorithm == Algorithm.DILITHIUM3:
            return self._signer.sign(message)
        else:
            return self._private_key.sign(message)

    def sign_json(self, data: dict[str, Any]) -> dict[str, Any]:
        """sign a JSON-serializable dict and return signed envelope."""
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        message = canonical.encode()
        signature = self.sign(message)

        return {
            "payload": data,
            "signature": signature.hex(),
            "algorithm": self._algorithm.value,
            "key_id": self._keypair.key_id,
            "signed_at": time.time(),
        }


class Verifier:
    """
    signature verifier - uses only public key.

    for hybrid mode, verifies BOTH signatures. fails if EITHER is invalid.
    """

    def __init__(
        self,
        public_key: bytes,
        algorithm: Algorithm,
        ed25519_public: bytes | None = None,
        dilithium_public: bytes | None = None,
    ):
        self._public_key = public_key
        self._algorithm = algorithm
        self._ed25519_public = ed25519_public
        self._dilithium_public = dilithium_public

        _require_liboqs(algorithm)

        if algorithm == Algorithm.HYBRID_ED25519_DILITHIUM3:
            if ed25519_public is None or dilithium_public is None:
                raise ValueError("hybrid mode requires both ed25519_public and dilithium_public")
            import oqs
            self._ed_public = ed25519.Ed25519PublicKey.from_public_bytes(ed25519_public)
            self._dil_verifier = oqs.Signature("Dilithium3")
        elif algorithm == Algorithm.DILITHIUM3:
            import oqs
            self._verifier = oqs.Signature("Dilithium3")
        elif algorithm == Algorithm.ED25519:
            self._ed_public = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
        else:
            raise ValueError(f"unsupported algorithm: {algorithm}")

    @property
    def algorithm(self) -> Algorithm:
        return self._algorithm

    def verify(self, message: bytes, signature: bytes) -> bool:
        """
        verify a signature.

        for hybrid mode, verifies BOTH signatures. returns False if either fails.
        """
        try:
            if self._algorithm == Algorithm.HYBRID_ED25519_DILITHIUM3:
                # parse length-prefixed signature
                if len(signature) < 4:
                    return False
                ed_len = int.from_bytes(signature[:4], 'big')
                if len(signature) < 4 + ed_len:
                    return False
                ed_sig = signature[4:4 + ed_len]
                dil_sig = signature[4 + ed_len:]

                # verify ed25519
                try:
                    self._ed_public.verify(ed_sig, message)
                except Exception:
                    logger.debug("hybrid verification: ed25519 signature invalid")
                    return False

                # verify dilithium - BOTH must pass
                if not self._dil_verifier.verify(message, dil_sig, self._dilithium_public):
                    logger.debug("hybrid verification: dilithium signature invalid")
                    return False

                return True

            elif self._algorithm == Algorithm.DILITHIUM3:
                return self._verifier.verify(message, signature, self._public_key)
            else:
                self._ed_public.verify(signature, message)
                return True
        except Exception as e:
            logger.debug(f"signature verification failed: {e}")
            return False

    def verify_json(self, signed_envelope: dict[str, Any]) -> tuple[bool, str]:
        """verify a signed JSON envelope."""
        try:
            payload = signed_envelope["payload"]
            signature = bytes.fromhex(signed_envelope["signature"])
            envelope_algo = signed_envelope.get("algorithm")

            # verify algorithm matches
            if envelope_algo and envelope_algo != self._algorithm.value:
                return False, f"algorithm mismatch: expected {self._algorithm.value}, got {envelope_algo}"

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
    def from_keypair(cls, keypair: KeyPair) -> "Verifier":
        """create verifier from keypair (uses public key only)."""
        return cls(
            keypair.public_key,
            keypair.algorithm,
            ed25519_public=keypair.ed25519_public,
            dilithium_public=keypair.dilithium_public,
        )


class SignedLogChain:
    """
    chain of signed log entries.

    each entry includes hash of previous entry, creating tamper-evident chain.
    """

    def __init__(self, signer: Signer):
        self._signer = signer
        self._chain: list[dict[str, Any]] = []
        self._prev_hash: str = "genesis"

    def _compute_hash(self, data: dict[str, Any]) -> str:
        """compute hash of entry."""
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

    def verify_chain(self, verifier: Verifier) -> tuple[bool, str]:
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


def get_recommended_algorithm(pq_required: bool) -> Algorithm:
    """
    get recommended algorithm based on requirements.

    if pq_required=True and liboqs missing, raises PQUnavailableError.
    """
    if pq_required:
        if not _liboqs_available():
            raise PQUnavailableError(
                "Post-quantum crypto required but liboqs-python not installed. "
                "Either install liboqs-python or set pq_crypto=false in config."
            )
        return Algorithm.DILITHIUM3
    else:
        return Algorithm.ED25519


def algorithm_from_config(config_value: str) -> Algorithm:
    """
    parse algorithm from config string value.

    raises ValueError for invalid values.
    raises PQUnavailableError if PQ algorithm selected but unavailable.
    """
    value = config_value.lower().strip()

    if value == "ed25519":
        return Algorithm.ED25519
    elif value == "dilithium3":
        _require_liboqs(Algorithm.DILITHIUM3)
        return Algorithm.DILITHIUM3
    elif value in ("hybrid", "hybrid_ed25519_dilithium3"):
        _require_liboqs(Algorithm.HYBRID_ED25519_DILITHIUM3)
        return Algorithm.HYBRID_ED25519_DILITHIUM3
    else:
        raise ValueError(
            f"invalid signature_algorithm: {config_value}. "
            "valid values: ed25519, dilithium3, hybrid"
        )


def liboqs_available() -> bool:
    """public API to check liboqs availability."""
    return _liboqs_available()


# backwards compatibility aliases (deprecated)
PQKeyPair = KeyPair
PQSigner = Signer
PQVerifier = Verifier
