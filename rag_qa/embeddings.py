"""Embedding client abstractions and Gemini implementation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Protocol, Sequence


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


class EmbeddingClient(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class GeminiEmbedder:
    api_key: str
    model: str
    dimensions: int
    batch_size: int = 32
    max_retries: int = 3
    _client: object | None = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(
                    "google-genai is not installed. Run 'uv sync' and try again."
                ) from exc
            if not self.api_key:
                raise EmbeddingError(
                    "GOOGLE_API_KEY is not set. Add it to .env before running embeddings."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _embed_batch(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        attempt = 0
        while True:
            attempt += 1
            try:
                client = self._get_client()
                config = types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self.dimensions
                )
                result = client.models.embed_content(
                    model=self.model,
                    contents=list(texts),
                    config=config,
                )
                vectors: list[list[float]] = []
                embeddings = getattr(result, "embeddings", None) or []
                for emb in embeddings:
                    values = list(getattr(emb, "values", emb) or [])
                    if len(values) != self.dimensions:
                        raise EmbeddingError(
                            f"Embedding dimension mismatch for model '{self.model}': expected "
                            f"{self.dimensions}, got {len(values)}. Check EMBEDDING_DIMENSIONS."
                        )
                    vectors.append([float(v) for v in values])
                if len(vectors) != len(texts):
                    raise EmbeddingError(
                        f"Embedding count mismatch for model '{self.model}': expected "
                        f"{len(texts)} vectors, got {len(vectors)}. Retry ingestion."
                    )
                return vectors
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize provider errors
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                message = str(exc)
                retryable = status in {429, 500, 502, 503, 504} or "429" in message
                if isinstance(status, int) and 500 <= status <= 599:
                    retryable = True
                if retryable and attempt < self.max_retries:
                    delay = (2 ** (attempt - 1)) + random.random() * 0.25
                    time.sleep(delay)
                    continue
                action = (
                    "Check GOOGLE_API_KEY, model access, and network connectivity, then retry."
                )
                raise EmbeddingError(
                    f"Failed to embed {len(texts)} text(s) with '{self.model}'. {action}"
                ) from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]
