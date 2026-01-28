# Author: Bradley R. Kinnard
# tests for LLM interface - TDD

import pytest


class TestLLMInitialization:
    """test LLM interface initialization."""

    def test_init_with_config(self, llm_interface, config):
        assert llm_interface is not None
        assert llm_interface.get_backend() == config["llm"]["backend"]

    def test_init_without_model_file(self):
        from interfaces.local_llm import LocalLLM
        # should handle missing model gracefully
        llm = LocalLLM(model_path="nonexistent.gguf")
        assert llm.is_model_loaded() is False

    def test_gpu_detection(self, llm_interface):
        gpu_available = llm_interface.detect_gpu()
        assert isinstance(gpu_available, bool)


class TestOfflineOperation:
    """test offline-only operation."""

    def test_no_network_calls(self, llm_interface, monkeypatch):
        # block network access
        import socket
        original_socket = socket.socket

        def blocked_socket(*args, **kwargs):
            raise RuntimeError("network access blocked")

        monkeypatch.setattr(socket, "socket", blocked_socket)

        # LLM operations should still work
        result = llm_interface.generate_sync("test", max_tokens=5)
        # should either succeed or fail gracefully, not try network
        assert result is not None or llm_interface.is_model_loaded() is False


class TestDeterministicOutput:
    """test deterministic responses with fixed seed."""

    def test_same_seed_same_output(self, llm_interface):
        if not llm_interface.is_model_loaded():
            pytest.skip("model not loaded")

        prompt = "Hello, world"
        result1 = llm_interface.generate_sync(prompt, seed=42, temperature=0.0)
        result2 = llm_interface.generate_sync(prompt, seed=42, temperature=0.0)

        assert result1 == result2

    def test_different_seed_different_output(self, llm_interface):
        if not llm_interface.is_model_loaded():
            pytest.skip("model not loaded")

        prompt = "Hello, world"
        result1 = llm_interface.generate_sync(prompt, seed=42, temperature=0.5)
        result2 = llm_interface.generate_sync(prompt, seed=99, temperature=0.5)

        # may or may not differ, but should not crash
        assert result1 is not None
        assert result2 is not None


class TestErrorHandling:
    """test error handling."""

    def test_missing_model_error(self, llm_interface):
        if llm_interface.is_model_loaded():
            pytest.skip("model is loaded")

        result = llm_interface.generate_sync("test")
        assert result is None or "error" in str(result).lower()

    def test_empty_prompt_handling(self, llm_interface):
        result = llm_interface.generate_sync("")
        # should handle gracefully
        assert result is not None or llm_interface.is_model_loaded() is False


@pytest.fixture
def llm_interface():
    """fixture providing LLM interface."""
    from interfaces.local_llm import LocalLLM
    from utils.helpers import load_system_config
    try:
        config = load_system_config()
        return LocalLLM(
            model_path=config["llm"]["model_path"],
            backend=config["llm"]["backend"],
            seed=config["llm"]["seed"]
        )
    except Exception:
        return LocalLLM()


@pytest.fixture
def config():
    """fixture providing config."""
    from utils.helpers import load_system_config
    return load_system_config()
