# Author: Bradley R. Kinnard
# ollama LLM adapter - local inference via ollama API

import json
import time
from typing import Any
from dataclasses import dataclass, field
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from utils.helpers import get_logger


logger = get_logger(__name__)


@dataclass
class OllamaConfig:
    """configuration for ollama LLM connection."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 512
    context_size: int = 4096
    timeout: int = 60
    num_thread: int = 4


@dataclass
class GenerationResult:
    """structured result from LLM generation."""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_ms: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)


class OllamaLLM:
    """
    ollama LLM adapter for local inference.
    provides deterministic generation with seeding support.
    """

    def __init__(self, config: OllamaConfig | None = None):
        self._config = config or OllamaConfig()
        self._connected = False
        self._available_models: list[str] = []

        self._check_connection()

    def _check_connection(self) -> None:
        """verify ollama server is reachable."""
        try:
            req = Request(f"{self._config.base_url}/api/tags")
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                self._available_models = [m["name"] for m in data.get("models", [])]
                self._connected = True
                logger.info(f"connected to ollama, {len(self._available_models)} models available")
        except (URLError, HTTPError, TimeoutError) as e:
            logger.warning(f"ollama connection failed: {e}")
            self._connected = False

    def is_connected(self) -> bool:
        """check if ollama server is reachable."""
        return self._connected

    def list_models(self) -> list[str]:
        """return list of available models."""
        return self._available_models.copy()

    def has_model(self, model_name: str) -> bool:
        """check if a specific model is available."""
        return any(model_name in m for m in self._available_models)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        raw: bool = False
    ) -> GenerationResult | None:
        """
        generate text from prompt using ollama API.
        supports system prompts and deterministic seeding.
        """
        if not self._connected:
            logger.error("ollama not connected")
            return None

        if not prompt:
            return GenerationResult(text="", model=self._config.model)

        start = time.perf_counter()

        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self._config.max_tokens,
                "seed": seed if seed is not None else self._config.seed,
                "num_ctx": self._config.context_size,
                "num_thread": self._config.num_thread,
            }
        }

        if system:
            payload["system"] = system

        try:
            req = Request(
                f"{self._config.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urlopen(req, timeout=self._config.timeout) as resp:
                data = json.loads(resp.read().decode())

            elapsed = (time.perf_counter() - start) * 1000

            result = GenerationResult(
                text=data.get("response", "").strip(),
                model=data.get("model", self._config.model),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                elapsed_ms=elapsed,
                raw_response=data if raw else {}
            )

            logger.debug(f"generated {len(result.text)} chars in {elapsed:.1f}ms")
            return result

        except (URLError, HTTPError, TimeoutError) as e:
            logger.error(f"generation failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"invalid response from ollama: {e}")
            return None

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None
    ) -> GenerationResult | None:
        """
        chat-style generation with message history.
        messages format: [{"role": "user|assistant|system", "content": "..."}]
        """
        if not self._connected:
            logger.error("ollama not connected")
            return None

        if not messages:
            return GenerationResult(text="", model=self._config.model)

        start = time.perf_counter()

        payload = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self._config.max_tokens,
                "seed": seed if seed is not None else self._config.seed,
                "num_ctx": self._config.context_size,
                "num_thread": self._config.num_thread,
            }
        }

        try:
            req = Request(
                f"{self._config.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urlopen(req, timeout=self._config.timeout) as resp:
                data = json.loads(resp.read().decode())

            elapsed = (time.perf_counter() - start) * 1000
            message = data.get("message", {})

            result = GenerationResult(
                text=message.get("content", "").strip(),
                model=data.get("model", self._config.model),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                elapsed_ms=elapsed,
            )

            logger.debug(f"chat generated {len(result.text)} chars in {elapsed:.1f}ms")
            return result

        except (URLError, HTTPError, TimeoutError) as e:
            logger.error(f"chat failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"invalid response from ollama: {e}")
            return None

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        seed: int | None = None
    ) -> dict[str, Any] | None:
        """
        generate structured JSON output.
        parses response as JSON, returns None on parse failure.
        """
        json_system = (system or "") + "\n\nRespond ONLY with valid JSON, no markdown or extra text."

        result = self.generate(
            prompt=prompt,
            system=json_system.strip(),
            seed=seed,
            temperature=0.0  # always deterministic for JSON
        )

        if not result or not result.text:
            return None

        text = result.text.strip()

        # strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # remove first and last line if they're fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"failed to parse JSON response: {e}")
            logger.debug(f"raw response: {result.text}")
            return None

    def set_model(self, model: str) -> bool:
        """switch to a different model."""
        if not self.has_model(model):
            logger.warning(f"model {model} not found")
            return False

        self._config.model = model
        logger.info(f"switched to model: {model}")
        return True

    def get_config(self) -> OllamaConfig:
        """return current configuration."""
        return self._config

    def embed(self, text: str) -> list[float] | None:
        """
        get text embedding using ollama.
        requires an embedding-capable model like nomic-embed-text.
        """
        if not self._connected:
            return None

        payload = {
            "model": self._config.model,
            "input": text
        }

        try:
            req = Request(
                f"{self._config.base_url}/api/embed",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urlopen(req, timeout=self._config.timeout) as resp:
                data = json.loads(resp.read().decode())

            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            return None

        except (URLError, HTTPError, TimeoutError) as e:
            logger.error(f"embedding failed: {e}")
            return None
