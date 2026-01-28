# Author: Bradley R. Kinnard
# persistent memory - async I/O with optional homomorphic encryption

import json
import asyncio
from pathlib import Path
from typing import Any

import aiofiles

from utils.helpers import get_logger


logger = get_logger(__name__)


class PersistentMemory:
    """
    persistent belief storage with async I/O support.
    optional homomorphic encryption for privacy-preserving operations.
    """

    def __init__(self, enable_he: bool = False):
        self._beliefs: dict[str, dict[str, Any]] = {}
        self._enable_he = enable_he
        self._he_context = None
        self._encrypted_store: dict[str, Any] = {}

        if enable_he:
            self._init_he()

    def _init_he(self) -> None:
        """initialize homomorphic encryption context."""
        try:
            import tenseal as ts
            self._he_context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self._he_context.global_scale = 2**40
            self._he_context.generate_galois_keys()
            logger.info("initialized tenseal HE context")
        except ImportError:
            logger.warning("tenseal not available, HE disabled")
            self._enable_he = False

    def store_belief(self, belief: dict[str, Any]) -> None:
        """store a belief in memory."""
        if "id" not in belief:
            raise ValueError("belief must have 'id' field")

        belief_id = belief["id"]
        self._beliefs[belief_id] = belief.copy()
        logger.debug(f"stored belief {belief_id}")

    def get_belief(self, belief_id: str) -> dict[str, Any] | None:
        """retrieve a belief by id."""
        return self._beliefs.get(belief_id)

    def list_beliefs(self) -> list[dict[str, Any]]:
        """list all stored beliefs."""
        return list(self._beliefs.values())

    def delete_belief(self, belief_id: str) -> bool:
        """delete a belief by id."""
        if belief_id in self._beliefs:
            del self._beliefs[belief_id]
            logger.debug(f"deleted belief {belief_id}")
            return True
        return False

    def save_to_disk(self, path: Path | str) -> None:
        """save beliefs to disk as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self._beliefs, f, indent=2)

        logger.info(f"saved {len(self._beliefs)} beliefs to {path}")

    def load_from_disk(self, path: Path | str) -> None:
        """load beliefs from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"corrupted or invalid JSON: {e}")

        self._beliefs = data
        logger.info(f"loaded {len(self._beliefs)} beliefs from {path}")

    async def async_store_belief(self, belief: dict[str, Any]) -> None:
        """async store a belief."""
        if "id" not in belief:
            raise ValueError("belief must have 'id' field")

        # simulate async operation
        await asyncio.sleep(0)
        self._beliefs[belief["id"]] = belief.copy()
        logger.debug(f"async stored belief {belief['id']}")

    async def async_get_belief(self, belief_id: str) -> dict[str, Any] | None:
        """async retrieve a belief."""
        await asyncio.sleep(0)
        return self._beliefs.get(belief_id)

    async def async_save_to_disk(self, path: Path | str) -> None:
        """async save beliefs to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(self._beliefs, indent=2)
        async with aiofiles.open(path, "w") as f:
            await f.write(content)

        logger.info(f"async saved {len(self._beliefs)} beliefs to {path}")

    async def async_load_from_disk(self, path: Path | str) -> None:
        """async load beliefs from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        async with aiofiles.open(path, "r") as f:
            content = await f.read()

        try:
            self._beliefs = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"corrupted or invalid JSON: {e}")

        logger.info(f"async loaded {len(self._beliefs)} beliefs from {path}")

    # homomorphic encryption methods

    def encrypt_belief(self, belief: dict[str, Any]) -> dict[str, Any]:
        """encrypt a belief using HE."""
        if not self._enable_he or self._he_context is None:
            raise RuntimeError("HE not enabled")

        import tenseal as ts

        vector = belief.get("vector", [belief.get("confidence", 0.5)])
        encrypted = ts.ckks_vector(self._he_context, vector)

        return {
            "id": belief["id"],
            "ciphertext": encrypted.serialize(),
            "metadata": {k: v for k, v in belief.items() if k not in ["vector", "confidence"]}
        }

    def store_encrypted(self, belief: dict[str, Any]) -> None:
        """store an encrypted belief."""
        if not self._enable_he:
            raise RuntimeError("HE not enabled")

        encrypted = self.encrypt_belief(belief)
        self._encrypted_store[belief["id"]] = encrypted
        logger.debug(f"stored encrypted belief {belief['id']}")

    def update_confidence_encrypted(
        self,
        belief_id: str,
        delta: float
    ) -> dict[str, Any]:
        """update confidence on encrypted belief (homomorphic add)."""
        if not self._enable_he or self._he_context is None:
            raise RuntimeError("HE not enabled")

        if belief_id not in self._encrypted_store:
            raise KeyError(f"encrypted belief not found: {belief_id}")

        import tenseal as ts

        encrypted = self._encrypted_store[belief_id]
        ciphertext = ts.ckks_vector_from(self._he_context, encrypted["ciphertext"])

        # homomorphic addition
        ciphertext += delta

        encrypted["ciphertext"] = ciphertext.serialize()
        self._encrypted_store[belief_id] = encrypted

        logger.debug(f"updated encrypted belief {belief_id} with delta {delta}")
        return {"success": True, "belief_id": belief_id}

    def decrypt_belief(self, belief_id: str) -> dict[str, Any]:
        """decrypt and return belief."""
        if not self._enable_he or self._he_context is None:
            raise RuntimeError("HE not enabled")

        if belief_id not in self._encrypted_store:
            raise KeyError(f"encrypted belief not found: {belief_id}")

        import tenseal as ts

        encrypted = self._encrypted_store[belief_id]
        ciphertext = ts.ckks_vector_from(self._he_context, encrypted["ciphertext"])

        decrypted_vector = ciphertext.decrypt()

        return {
            "id": belief_id,
            "confidence": decrypted_vector[0],
            "vector": decrypted_vector,
            **encrypted.get("metadata", {})
        }
