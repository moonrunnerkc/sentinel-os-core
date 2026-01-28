# Author: Bradley R. Kinnard
# tests for memory system - TDD

import pytest
import asyncio
import time
import tempfile
import os
from pathlib import Path


class TestMemoryStorage:
    """test memory storage operations."""

    def test_store_belief(self, memory):
        belief = {"id": "m1", "content": "test", "confidence": 0.8}
        memory.store_belief(belief)

        retrieved = memory.get_belief("m1")
        assert retrieved["content"] == "test"

    def test_store_rejects_invalid_belief(self, memory):
        with pytest.raises(ValueError):
            memory.store_belief({})  # missing required fields

    def test_update_existing_belief(self, memory):
        belief = {"id": "m2", "content": "original", "confidence": 0.5}
        memory.store_belief(belief)

        updated = {"id": "m2", "content": "updated", "confidence": 0.9}
        memory.store_belief(updated)

        retrieved = memory.get_belief("m2")
        assert retrieved["content"] == "updated"


class TestMemoryRetrieval:
    """test memory retrieval operations."""

    def test_retrieve_existing_belief(self, memory):
        belief = {"id": "r1", "content": "retrievable", "confidence": 0.7}
        memory.store_belief(belief)

        result = memory.get_belief("r1")
        assert result is not None
        assert result["id"] == "r1"

    def test_retrieve_nonexistent_returns_none(self, memory):
        result = memory.get_belief("nonexistent")
        assert result is None

    def test_list_all_beliefs(self, memory):
        for i in range(5):
            memory.store_belief({"id": f"list{i}", "content": f"belief {i}", "confidence": 0.5})

        all_beliefs = memory.list_beliefs()
        assert len(all_beliefs) >= 5


class TestPersistence:
    """test memory persistence across sessions."""

    def test_persist_and_reload(self, memory, tmp_path):
        belief = {"id": "persist1", "content": "persistent", "confidence": 0.6}
        memory.store_belief(belief)

        # save to disk
        save_path = tmp_path / "beliefs.json"
        memory.save_to_disk(save_path)

        # create new memory and load
        from memory.persistent_memory import PersistentMemory
        new_memory = PersistentMemory()
        new_memory.load_from_disk(save_path)

        retrieved = new_memory.get_belief("persist1")
        assert retrieved is not None
        assert retrieved["content"] == "persistent"

    def test_corruption_handling(self, memory, tmp_path):
        # write corrupted data
        corrupt_path = tmp_path / "corrupt.json"
        corrupt_path.write_text("not valid json {{{")

        with pytest.raises(ValueError, match="corrupted|invalid|parse"):
            memory.load_from_disk(corrupt_path)


class TestAsyncIO:
    """test async I/O operations."""

    @pytest.mark.asyncio
    async def test_async_store(self, async_memory, tmp_path):
        belief = {"id": "async1", "content": "async test", "confidence": 0.8}
        await async_memory.async_store_belief(belief)

        retrieved = await async_memory.async_get_belief("async1")
        assert retrieved["content"] == "async test"

    @pytest.mark.asyncio
    async def test_async_save_performance(self, async_memory, tmp_path):
        # store 100 beliefs
        for i in range(100):
            belief = {"id": f"perf{i}", "content": f"perf test {i}", "confidence": 0.5}
            await async_memory.async_store_belief(belief)

        # measure async save time
        save_path = tmp_path / "async_perf.json"
        start = time.time()
        await async_memory.async_save_to_disk(save_path)
        async_time = time.time() - start

        # should be fast for 100 beliefs
        assert async_time < 1.0


class TestHomomorphicEncryption:
    """test homomorphic encryption for beliefs."""

    def test_encrypt_belief(self, he_memory, config):
        if not config.get("use_homomorphic_enc", False):
            pytest.skip("HE disabled in config")

        belief = {"id": "he1", "content": "secret", "confidence": 0.75, "vector": [0.1, 0.2, 0.3]}
        encrypted = he_memory.encrypt_belief(belief)

        assert encrypted is not None
        assert "ciphertext" in encrypted

    def test_encrypted_operations(self, he_memory, config):
        if not config.get("use_homomorphic_enc", False):
            pytest.skip("HE disabled in config")

        # store encrypted
        belief = {"id": "he2", "content": "compute", "confidence": 0.5, "vector": [1.0, 2.0, 3.0]}
        he_memory.store_encrypted(belief)

        # perform operation on ciphertext
        result = he_memory.update_confidence_encrypted("he2", delta=0.1)

        # verify without decrypting (check operation succeeded)
        assert result["success"] is True

    def test_encrypted_accuracy(self, he_memory, config):
        """verify >95% accuracy vs plaintext operations."""
        if not config.get("use_homomorphic_enc", False):
            pytest.skip("HE disabled in config")

        correct = 0
        for i in range(100):
            vector = [float(i), float(i+1), float(i+2)]
            belief = {"id": f"acc{i}", "confidence": 0.5, "vector": vector}

            # plaintext operation
            plain_result = 0.5 + 0.1  # simple add

            # encrypted operation
            he_memory.store_encrypted(belief)
            he_result = he_memory.update_confidence_encrypted(f"acc{i}", delta=0.1)
            decrypted = he_memory.decrypt_belief(f"acc{i}")

            # check if close enough
            if abs(decrypted["confidence"] - plain_result) < 0.01:
                correct += 1

        accuracy = correct / 100
        assert accuracy >= 0.95, f"HE accuracy {accuracy:.2%} < 95%"


@pytest.fixture
def memory():
    """fixture providing fresh persistent memory."""
    from memory.persistent_memory import PersistentMemory
    return PersistentMemory()


@pytest.fixture
def async_memory():
    """fixture providing async-capable memory."""
    from memory.persistent_memory import PersistentMemory
    return PersistentMemory()


@pytest.fixture
def he_memory():
    """fixture providing HE-enabled memory."""
    from memory.persistent_memory import PersistentMemory
    return PersistentMemory(enable_he=True)


@pytest.fixture
def config():
    """fixture providing security config."""
    from utils.helpers import load_security_rules
    try:
        return load_security_rules()
    except FileNotFoundError:
        return {"use_homomorphic_enc": False}
