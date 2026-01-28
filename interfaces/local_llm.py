# Author: Bradley R. Kinnard
# local LLM interface - offline inference with llama.cpp

from pathlib import Path

from utils.helpers import get_logger


logger = get_logger(__name__)


class LocalLLM:
    """
    local LLM interface using llama.cpp.
    supports offline operation with deterministic seeding.
    """

    def __init__(
        self,
        model_path: str | Path = "",
        backend: str = "llama-cpp",
        seed: int = 42,
        gpu_layers: int = 0
    ):
        self._model_path = Path(model_path) if model_path else None
        self._backend = backend
        self._seed = seed
        self._gpu_layers = gpu_layers
        self._model = None
        self._loaded = False

        if self._model_path and self._model_path.exists():
            self._load_model()
        else:
            logger.warning(f"model not found: {model_path}")

    def _load_model(self) -> None:
        """load the LLM model."""
        try:
            import llama_cpp

            # auto-detect GPU
            detected_gpu = self.detect_gpu()
            layers = self._gpu_layers
            if detected_gpu and layers == 0:
                layers = 32
                logger.info(f"GPU detected: offloading {layers} layers")

            self._model = llama_cpp.Llama(
                model_path=str(self._model_path),
                n_gpu_layers=layers,
                seed=self._seed,
                verbose=False
            )
            self._loaded = True
            logger.info(f"loaded model: {self._model_path.name}")

        except ImportError:
            logger.warning("llama-cpp-python not installed")
            self._loaded = False
        except Exception as e:
            logger.error(f"failed to load model: {e}")
            self._loaded = False

    def is_model_loaded(self) -> bool:
        """check if model is loaded."""
        return self._loaded

    def get_backend(self) -> str:
        """return backend name."""
        return self._backend

    def detect_gpu(self) -> bool:
        """detect if GPU offload is available."""
        try:
            import llama_cpp
            if hasattr(llama_cpp, "llama_supports_gpu_offload"):
                return llama_cpp.llama_supports_gpu_offload()
            return False
        except ImportError:
            return False

    def generate_sync(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None
    ) -> str | None:
        """synchronous text generation."""
        if not self._loaded:
            logger.warning("model not loaded, cannot generate")
            return None

        if not prompt:
            return ""

        try:
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed if seed is not None else self._seed
            )
            text = result["choices"][0]["text"]
            logger.debug(f"generated {len(text)} chars")
            return text

        except Exception as e:
            logger.error(f"generation failed: {e}")
            return None

    def generate_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None
    ) -> str | None:
        """generate with system and user prompts."""
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        return self.generate_sync(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed
        )

    def get_embedding(self, text: str) -> list[float] | None:
        """get text embedding (if supported)."""
        if not self._loaded:
            return None

        try:
            # llama-cpp may not support embeddings directly
            # this is a placeholder for models that do
            logger.warning("embedding not implemented for this backend")
            return None
        except Exception as e:
            logger.error(f"embedding failed: {e}")
            return None
