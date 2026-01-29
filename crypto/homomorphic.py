# Author: Bradley R. Kinnard
# homomorphic encryption operations for private belief computation
# requires TenSEAL - no mock mode, fails explicitly if unavailable

from dataclasses import dataclass
from typing import Any
import time

from utils.helpers import get_logger

logger = get_logger(__name__)


def _check_tenseal():
    """verify TenSEAL is available, raise ImportError if not."""
    try:
        import tenseal  # noqa: F401
        return True
    except ImportError:
        return False


def tenseal_available() -> bool:
    """public API to check TenSEAL availability."""
    return _check_tenseal()


def validate_he_config(config: dict) -> None:
    """
    validate HE configuration at startup.

    raises HEUnavailableError if HE is enabled but TenSEAL is missing.
    this is called at startup to fail fast, not silently.
    """
    features = config.get("features", {})
    he_enabled = features.get("homomorphic_encryption", False)

    if he_enabled and not _check_tenseal():
        raise HEUnavailableError(
            "FATAL: homomorphic_encryption enabled in config but tenseal not installed. "
            "Either install tenseal (pip install tenseal) or set "
            "features.homomorphic_encryption: false in config/system_config.yaml. "
            "No silent fallback is permitted."
        )

    if he_enabled:
        logger.info("homomorphic encryption: enabled (tenseal available)")
    else:
        logger.debug("homomorphic encryption: disabled in config")


@dataclass
class HEContext:
    """homomorphic encryption context wrapper."""
    context_bytes: bytes
    scheme: str
    poly_modulus_degree: int
    coeff_mod_bit_sizes: list[int]
    scale: float
    created_at: float


class HEUnavailableError(ImportError):
    """raised when HE operations are attempted without TenSEAL."""
    pass


class HomomorphicEngine:
    """
    homomorphic encryption engine for privacy-preserving belief operations.

    requires TenSEAL to be installed. raises HEUnavailableError if not.

    supports:
    - encrypted belief confidence storage
    - encrypted addition/subtraction of confidence deltas
    - encrypted aggregation across beliefs
    - decryption for final readout

    uses CKKS scheme for approximate arithmetic on real numbers.
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list[int] | None = None,
        scale: float = 2**40,
    ):
        if not _check_tenseal():
            raise HEUnavailableError(
                "Homomorphic encryption requires TenSEAL. "
                "Install with: pip install tenseal"
            )

        import tenseal as ts
        self._ts = ts

        self._poly_modulus_degree = poly_modulus_degree
        self._coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self._scale = scale

        self._context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self._poly_modulus_degree,
            coeff_mod_bit_sizes=self._coeff_mod_bit_sizes,
        )
        self._context.generate_galois_keys()
        self._context.global_scale = self._scale

        logger.info(
            f"initialized CKKS context: poly_mod={self._poly_modulus_degree}, "
            f"scale={self._scale}"
        )

    def encrypt_scalar(self, value: float) -> "EncryptedValue":
        """encrypt a single scalar value."""
        encrypted = self._ts.ckks_vector(self._context, [value])
        return EncryptedValue(
            ciphertext=encrypted,
            encrypted_at=time.time(),
        )

    def encrypt_vector(self, values: list[float]) -> "EncryptedValue":
        """encrypt a vector of values."""
        encrypted = self._ts.ckks_vector(self._context, values)
        return EncryptedValue(
            ciphertext=encrypted,
            encrypted_at=time.time(),
        )

    def decrypt_scalar(self, encrypted: "EncryptedValue") -> float:
        """decrypt to scalar."""
        result = encrypted.ciphertext.decrypt()
        return result[0]

    def decrypt_vector(self, encrypted: "EncryptedValue") -> list[float]:
        """decrypt to vector."""
        return encrypted.ciphertext.decrypt()

    def add(
        self,
        a: "EncryptedValue",
        b: "EncryptedValue",
    ) -> "EncryptedValue":
        """homomorphic addition of two encrypted values."""
        result = a.ciphertext + b.ciphertext
        return EncryptedValue(
            ciphertext=result,
            encrypted_at=time.time(),
        )

    def add_plain(
        self,
        encrypted: "EncryptedValue",
        plain: float | list[float],
    ) -> "EncryptedValue":
        """add plaintext to encrypted value."""
        if isinstance(plain, list):
            result = encrypted.ciphertext + plain
        else:
            result = encrypted.ciphertext + [plain]

        return EncryptedValue(
            ciphertext=result,
            encrypted_at=time.time(),
        )

    def multiply_plain(
        self,
        encrypted: "EncryptedValue",
        plain: float | list[float],
    ) -> "EncryptedValue":
        """multiply encrypted value by plaintext."""
        if isinstance(plain, list):
            result = encrypted.ciphertext * plain
        else:
            result = encrypted.ciphertext * [plain]

        return EncryptedValue(
            ciphertext=result,
            encrypted_at=time.time(),
        )

    def sum_encrypted(
        self,
        values: list["EncryptedValue"],
    ) -> "EncryptedValue":
        """sum multiple encrypted values."""
        if not values:
            return self.encrypt_scalar(0.0)

        result = values[0]
        for v in values[1:]:
            result = self.add(result, v)

        return result


@dataclass
class EncryptedValue:
    """wrapper for encrypted values."""
    ciphertext: Any  # ts.CKKSVector
    encrypted_at: float

    def size_bytes(self) -> int:
        """size of ciphertext in bytes."""
        if self.ciphertext is None:
            return 0
        return len(self.ciphertext.serialize())


class EncryptedBeliefStore:
    """
    encrypted belief storage using homomorphic encryption.

    stores belief confidences encrypted, allowing:
    - update confidence without decryption
    - aggregate confidence across beliefs
    - only decrypt for final readout
    """

    def __init__(self, engine: HomomorphicEngine):
        self._engine = engine
        self._beliefs: dict[str, EncryptedValue] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def store_belief(
        self,
        belief_id: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """store a belief with encrypted confidence."""
        encrypted = self._engine.encrypt_scalar(confidence)
        self._beliefs[belief_id] = encrypted
        self._metadata[belief_id] = metadata or {}

        logger.debug(f"stored encrypted belief {belief_id}")

    def update_confidence(
        self,
        belief_id: str,
        delta: float,
    ) -> None:
        """update belief confidence homomorphically (without decryption)."""
        if belief_id not in self._beliefs:
            raise KeyError(f"belief {belief_id} not found")

        current = self._beliefs[belief_id]
        updated = self._engine.add_plain(current, delta)
        self._beliefs[belief_id] = updated

        logger.debug(f"updated belief {belief_id} by delta {delta}")

    def get_confidence(self, belief_id: str) -> float:
        """decrypt and return belief confidence."""
        if belief_id not in self._beliefs:
            raise KeyError(f"belief {belief_id} not found")

        encrypted = self._beliefs[belief_id]
        return self._engine.decrypt_scalar(encrypted)

    def aggregate_confidence(self, belief_ids: list[str] | None = None) -> float:
        """compute aggregate (sum) of confidences."""
        if belief_ids is None:
            belief_ids = list(self._beliefs.keys())

        encrypted_values = [self._beliefs[bid] for bid in belief_ids if bid in self._beliefs]

        if not encrypted_values:
            return 0.0

        total = self._engine.sum_encrypted(encrypted_values)
        return self._engine.decrypt_scalar(total)

    def get_stats(self) -> dict[str, Any]:
        """get storage statistics."""
        total_size = sum(v.size_bytes() for v in self._beliefs.values())

        return {
            "n_beliefs": len(self._beliefs),
            "total_size_bytes": total_size,
        }
