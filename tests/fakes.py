"""Deterministic test doubles."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class FakeEmbedder:
    dimensions: int = 768
    document_calls: list[list[str]] = field(default_factory=list)
    query_calls: list[str] = field(default_factory=list)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = _tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = int.from_bytes(digest[i : i + 2], "little") % self.dimensions
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str):
        self.query_calls.append(text)
        return self._embed(text)
