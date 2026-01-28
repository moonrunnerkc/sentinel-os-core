# Author: Bradley R. Kinnard
# homomorphic encryption operations for private belief computation

from dataclasses import dataclass
from typing import Any
import time
import json

from utils.helpers import get_logger

logger = get_logger(__name__)


# check if TenSEAL is available
_TENSEAL_AVAILABLE = False
try:
    import tenseal as ts
    _TENSEAL_AVAILABLE = True
    logger.info("TenSEAL available - HE operations enabled")
except ImportError:
    logger.warning("TenSEAL not available - HE operations will use mock mode")


@dataclass
class HEContext:
    """homomorphic encryption context wrapper."""
    context_bytes: bytes
    scheme: str
    poly_modulus_degree: int
    coeff_mod_bit_sizes: list[int]
    scale: float
    created_at: float


class HomomorphicEngine:
    """
    homomorphic encryption engine for privacy-preserving belief operations.

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
        self._poly_modulus_degree = poly_modulus_degree
        self._coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self._scale = scale
        self._context = None
        self._secret_key = None
        self._public_key = None
        self._relin_keys = None
        self._galois_keys = None
        self._mock_mode = not _TENSEAL_AVAILABLE

        if not self._mock_mode:
            self._init_tenseal_context()
        else:
            logger.warning("HE engine running in mock mode")

    def _init_tenseal_context(self) -> None:
        """initialize TenSEAL CKKS context."""
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

    def is_mock_mode(self) -> bool:
        """check if running in mock mode."""
        return self._mock_mode

    def encrypt_scalar(self, value: float) -> "EncryptedValue":
        """encrypt a single scalar value."""
        if self._mock_mode:
            return EncryptedValue(
                ciphertext=None,
                mock_value=value,
                is_mock=True,
                encrypted_at=time.time(),
            )

        encrypted = ts.ckks_vector(self._context, [value])
        return EncryptedValue(
            ciphertext=encrypted,
            mock_value=None,
            is_mock=False,
            encrypted_at=time.time(),
        )

    def encrypt_vector(self, values: list[float]) -> "EncryptedValue":
        """encrypt a vector of values."""
        if self._mock_mode:
            return EncryptedValue(
                ciphertext=None,
                mock_value=values,
                is_mock=True,
                encrypted_at=time.time(),
            )

        encrypted = ts.ckks_vector(self._context, values)
        return EncryptedValue(
            ciphertext=encrypted,
            mock_value=None,
            is_mock=False,
            encrypted_at=time.time(),
        )

    def decrypt_scalar(self, encrypted: "EncryptedValue") -> float:
        """decrypt to scalar."""
        if encrypted.is_mock:
            v = encrypted.mock_value
            return v[0] if isinstance(v, list) else v

        result = encrypted.ciphertext.decrypt()
        return result[0]

    def decrypt_vector(self, encrypted: "EncryptedValue") -> list[float]:
        """decrypt to vector."""
        if encrypted.is_mock:
            v = encrypted.mock_value
            return v if isinstance(v, list) else [v]

        return encrypted.ciphertext.decrypt()

    def add(
        self,
        a: "EncryptedValue",
        b: "EncryptedValue",
    ) -> "EncryptedValue":
        """homomorphic addition of two encrypted values."""
        if a.is_mock and b.is_mock:
            # mock addition
            if isinstance(a.mock_value, list) and isinstance(b.mock_value, list):
                result = [x + y for x, y in zip(a.mock_value, b.mock_value)]
            elif isinstance(a.mock_value, list):
                result = [x + b.mock_value for x in a.mock_value]
            elif isinstance(b.mock_value, list):
                result = [a.mock_value + y for y in b.mock_value]
            else:
                result = a.mock_value + b.mock_value

            return EncryptedValue(
                ciphertext=None,
                mock_value=result,
                is_mock=True,
                encrypted_at=time.time(),
            )

        result = a.ciphertext + b.ciphertext
        return EncryptedValue(
            ciphertext=result,
            mock_value=None,
            is_mock=False,
            encrypted_at=time.time(),
        )

    def add_plain(
        self,
        encrypted: "EncryptedValue",
        plain: float | list[float],
    ) -> "EncryptedValue":
        """add plaintext to encrypted value."""
        if encrypted.is_mock:
            if isinstance(encrypted.mock_value, list) and isinstance(plain, list):
                result = [x + y for x, y in zip(encrypted.mock_value, plain)]
            elif isinstance(encrypted.mock_value, list):
                result = [x + plain for x in encrypted.mock_value]
            else:
                result = encrypted.mock_value + plain

            return EncryptedValue(
                ciphertext=None,
                mock_value=result,
                is_mock=True,
                encrypted_at=time.time(),
            )

        if isinstance(plain, list):
            result = encrypted.ciphertext + plain
        else:
            result = encrypted.ciphertext + [plain]

        return EncryptedValue(
            ciphertext=result,
            mock_value=None,
            is_mock=False,
            encrypted_at=time.time(),
        )

    def multiply_plain(
        self,
        encrypted: "EncryptedValue",
        plain: float | list[float],
    ) -> "EncryptedValue":
        """multiply encrypted value by plaintext."""
        if encrypted.is_mock:
            if isinstance(encrypted.mock_value, list) and isinstance(plain, list):
                result = [x * y for x, y in zip(encrypted.mock_value, plain)]
            elif isinstance(encrypted.mock_value, list):
                result = [x * plain for x in encrypted.mock_value]
            else:
                result = encrypted.mock_value * plain

            return EncryptedValue(
                ciphertext=None,
                mock_value=result,
                is_mock=True,
                encrypted_at=time.time(),
            )

        if isinstance(plain, list):
            result = encrypted.ciphertext * plain
        else:
            result = encrypted.ciphertext * [plain]

        return EncryptedValue(
            ciphertext=result,
            mock_value=None,
            is_mock=False,
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
    ciphertext: Any  # ts.CKKSVector or None
    mock_value: Any
    is_mock: bool
    encrypted_at: float

    def size_bytes(self) -> int:
        """estimate size of ciphertext."""
        if self.is_mock:
            return 8  # just a float
        if self.ciphertext is None:
            return 0
        # estimate based on poly modulus degree
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
            "is_mock_mode": self._engine.is_mock_mode(),
        }
