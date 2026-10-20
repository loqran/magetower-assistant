"""
Wrapper for ollama HTTP api key
    /api/chat
    /api/tags

standard lib only, no dependencies
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator

from .config import LLMConfig

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class OllamaError(RuntimeError):
    """Raised for connection problem or error response from ollama."""


class OllamaClient:
    def __init__(self, cfg: LLMConfig, timeout: float = 300.0):
        self.cfg = cfg
        self.base = cfg.host.rstrip("/")

        # generous timeout
        self.timeout = timeout

        # ollama loca, never route thru HTTP_PROXY from the env
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def close(self) -> None:
        pass  # nothing pooled, keeping so callers don't change if transport done

    def _open(
        self, path: str, payload: dict | None = None, timeout: float | None = None
    ):

        data = json.dumps(payload).encode() if payload is not None else None

        req = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
        )

        try:
            return self._opener.open(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise OllamaError(f"Ollama returned {e.code}: {body}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise OllamaError(
                f"Can't reach ollama at {self.cfg.host} ({getattr(e, 'reason', e)}). "
                "Is it running? try -> systemctl start ollama"
            ) from e

    # -- health

    def check(self, *extra_models: str) -> None:
        """Fail fast with usefull mesage if ollama is down or a model isn't pulled"""
        with self._open("/api/tags", timeout=5.0) as resp:
            models = json.load(resp).get("models", [])

        names = {m.get("name", "") for m in models}

        # granite3.3:8b should match itself, nomic-embed-text matches its :latest tag
        for wanted in (self.cfg.model, *extra_models):
            if wanted not in names and f"{wanted}:latest" not in names:
                raise OllamaError(
                    f"Model '{wanted}' isn't pulled. -> Run ollama pull {wanted}\n"
                    f"available: {', '.join(sorted(names)) or '(none)'}"
                )

    # -- embeddings
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed a batch of texts with an Ollama embedding model"""
        with self._open("/api/embed", {"model": model, "input": texts}) as resp:
            return json.load(resp)["embeddings"]

    # -- chat
    def stream_chat(self, messages: list[Message], **options) -> Iterator[str]:
        """Yield response text chuncks as they arrive (NDJSON); options override config"""
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.cfg.temperature,
                "num_ctx": self.cfg.num_ctx,
                **options,
            },
        }
        with self._open("/api/chat", payload) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise OllamaError(chunk["error"])
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield text
                if chunk.get("done"):
                    break
