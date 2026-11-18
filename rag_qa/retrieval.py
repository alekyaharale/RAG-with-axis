"""Hybrid retrieval over the local chunk index."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

from .config import Settings
from .embeddings import EmbeddingClient
from .index import SearchIndex, StoredChunk, load_index, IndexError as StoredIndexError


class IndexNotBuilt(RuntimeError):
    """Raised when retrieval is attempted before ingestion."""


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


@dataclass
class BM25Index:
    chunks: list[StoredChunk]
    doc_tokens: list[list[str]]
    doc_freq: dict[str, int]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, chunks: list[StoredChunk]) -> "BM25Index":
        doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        doc_freq: dict[str, int] = {}
        for tokens in doc_tokens:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        avgdl = (sum(len(t) for t in doc_tokens) / len(doc_tokens)) if doc_tokens else 0.0
        return cls(chunks=chunks, doc_tokens=doc_tokens, doc_freq=doc_freq, avgdl=avgdl)

    def scores(self, query: str) -> dict[str, float]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return {}
        n = len(self.chunks)
        scores: dict[str, float] = {}
        for chunk, tokens in zip(self.chunks, self.doc_tokens, strict=True):
            if not tokens:
                continue
            tf_map: dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1
            score = 0.0
            dl = len(tokens)
            for token in q_tokens:
                tf = tf_map.get(token)
                if not tf:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                score += idf * ((tf * (self.k1 + 1)) / denom)
            if score > 0:
                scores[chunk.chunk_id] = score
        return scores


@dataclass
class RetrievalResult:
    chunk: StoredChunk
    score: float

    def to_payload(self) -> dict[str, object]:
        return {
            "source": self.chunk.source,
            "heading": self.chunk.heading,
            "page": self.chunk.page,
            "chunk_id": self.chunk.chunk_id,
            "score": round(self.score, 6),
            "text": self.chunk.text,
        }


class Retriever:
    def __init__(self, index: SearchIndex):
        self.index = index
        self.chunks = index.all_chunks()
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.bm25 = BM25Index.build(self.chunks)

    def search(
        self,
        query: str,
        query_vector: list[float],
        *,
        top_k: int,
        min_similarity: float,
        max_context_chars: int,
    ) -> list[RetrievalResult]:
        if not query.strip() or not self.chunks:
            return []

        dense_scored = [
            (chunk, cosine_similarity(query_vector, chunk.vector)) for chunk in self.chunks
        ]
        dense_scored.sort(key=lambda item: (-item[1], item[0].source, item[0].ordinal))
        dense_candidates = [
            (chunk, score) for chunk, score in dense_scored[:20] if score >= min_similarity
        ]

        lexical_scores = self.bm25.scores(query)
        lexical_ranked = sorted(
            (
                (self.by_id[cid], score)
                for cid, score in lexical_scores.items()
                if cid in self.by_id
            ),
            key=lambda item: (-item[1], item[0].source, item[0].ordinal),
        )[:20]

        rrf_k = 60
        fused: dict[str, float] = {}
        for rank, (chunk, _score) in enumerate(dense_candidates, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (chunk, _score) in enumerate(lexical_ranked, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

        ranked_ids = sorted(
            fused,
            key=lambda cid: (
                -fused[cid],
                self.by_id[cid].source,
                self.by_id[cid].ordinal,
            ),
        )

        results: list[RetrievalResult] = []
        used_chars = 0
        for cid in ranked_ids:
            if len(results) >= top_k:
                break
            chunk = self.by_id[cid]
            if used_chars + len(chunk.text) > max_context_chars and results:
                continue
            results.append(RetrievalResult(chunk=chunk, score=fused[cid]))
            used_chars += len(chunk.text)
            if used_chars >= max_context_chars:
                break
        return results


class RetrievalService:
    def __init__(self, settings: Settings, embedder: EmbeddingClient):
        self.settings = settings
        self.embedder = embedder
        self._retriever: Retriever | None = None

    def _load(self) -> Retriever:
        if self._retriever is not None:
            return self._retriever
        try:
            index = load_index(self.settings.index_path)
        except StoredIndexError as exc:
            raise IndexNotBuilt(
                "The reference index has not been built. Run 'uv run rag-ingest' to create "
                f"'{self.settings.index_path}'."
            ) from exc
        self._retriever = Retriever(index)
        return self._retriever

    def search(self, query: str) -> list[RetrievalResult]:
        retriever = self._load()
        query_vector = self.embedder.embed_query(query)
        return retriever.search(
            query,
            query_vector,
            top_k=self.settings.top_k,
            min_similarity=self.settings.min_similarity,
            max_context_chars=self.settings.max_context_chars,
        )
