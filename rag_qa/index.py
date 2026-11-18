"""Index build/load with incremental reuse and atomic persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from .chunking import Chunk, chunk_document
from .documents import SourceDocument
from .embeddings import EmbeddingClient

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexManifest:
    schema_version: int
    embedding_model: str
    embedding_dimensions: int
    chunk_chars: int
    chunk_overlap: int


@dataclass
class StoredChunk:
    chunk_id: str
    source: str
    heading: str
    page: int | None
    start_char: int
    end_char: int
    text: str
    content_hash: str
    ordinal: int
    vector: list[float]

    @classmethod
    def from_chunk(cls, chunk: Chunk, vector: list[float]) -> "StoredChunk":
        return cls(vector=vector, **asdict(chunk))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredChunk":
        return cls(**data)


@dataclass
class DocumentIndex:
    content_hash: str
    chunks: list[StoredChunk]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentIndex":
        return cls(
            content_hash=data["content_hash"],
            chunks=[StoredChunk.from_dict(c) for c in data.get("chunks", [])],
        )


@dataclass
class SearchIndex:
    manifest: IndexManifest
    documents: dict[str, DocumentIndex]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": asdict(self.manifest),
            "documents": {name: doc.to_dict() for name, doc in self.documents.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchIndex":
        manifest = IndexManifest(**data["manifest"])
        documents = {
            name: DocumentIndex.from_dict(doc) for name, doc in data.get("documents", {}).items()
        }
        return cls(manifest=manifest, documents=documents)

    def all_chunks(self) -> list[StoredChunk]:
        chunks: list[StoredChunk] = []
        for name in sorted(self.documents):
            chunks.extend(self.documents[name].chunks)
        chunks.sort(key=lambda c: (c.source, c.ordinal))
        return chunks


class IndexError(ValueError):
    """Raised for index read/write or compatibility issues."""


def _settings_match(index: SearchIndex, settings: Any) -> bool:
    m = index.manifest
    return (
        m.schema_version == SCHEMA_VERSION
        and m.embedding_model == settings.embedding_model
        and m.embedding_dimensions == settings.embedding_dimensions
        and m.chunk_chars == settings.chunk_chars
        and m.chunk_overlap == settings.chunk_overlap
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                tmp_path = Path(handle.name)
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if attempt == 0:
                time.sleep(0.1)
                continue
            raise
        except Exception:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise


def save_index(index: SearchIndex, path: Path) -> None:
    _atomic_write_json(path, index.to_dict())


def load_index(path: Path) -> SearchIndex:
    if not path.exists():
        raise IndexError(f"Index file '{path}' was not found.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexError(
            f"Index file '{path}' is not valid JSON. Run 'uv run rag-ingest' to rebuild it."
        ) from exc
    try:
        index = SearchIndex.from_dict(data)
    except Exception as exc:  # noqa: BLE001 - schema validation
        raise IndexError(
            f"Index file '{path}' has an unsupported format. Run 'uv run rag-ingest' to rebuild it."
        ) from exc
    if index.manifest.schema_version != SCHEMA_VERSION:
        raise IndexError(
            f"Index schema version {index.manifest.schema_version} is not supported. "
            "Run 'uv run rag-ingest' to rebuild it."
        )
    return index


@dataclass
class BuildStats:
    documents: int
    chunks: int
    reused: int
    embedded: int


def build_index(
    documents: list[SourceDocument],
    settings: Any,
    embedder: EmbeddingClient,
    existing: SearchIndex | None = None,
) -> tuple[SearchIndex, BuildStats]:
    can_reuse = existing is not None and _settings_match(existing, settings)
    existing_docs = existing.documents if can_reuse and existing else {}

    new_documents: dict[str, DocumentIndex] = {}
    reused_chunks = 0
    embedded_chunks = 0
    total_chunks = 0

    for document in sorted(documents, key=lambda d: d.name):
        prior = existing_docs.get(document.name)
        if prior and prior.content_hash == document.content_hash:
            new_documents[document.name] = prior
            reused_chunks += len(prior.chunks)
            total_chunks += len(prior.chunks)
            continue

        chunks = chunk_document(
            document, chunk_chars=settings.chunk_chars, chunk_overlap=settings.chunk_overlap
        )
        vectors = embedder.embed_documents([chunk.text for chunk in chunks]) if chunks else []
        stored = [StoredChunk.from_chunk(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)]
        new_documents[document.name] = DocumentIndex(
            content_hash=document.content_hash, chunks=stored
        )
        embedded_chunks += len(stored)
        total_chunks += len(stored)

    index = SearchIndex(
        manifest=IndexManifest(
            schema_version=SCHEMA_VERSION,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            chunk_chars=settings.chunk_chars,
            chunk_overlap=settings.chunk_overlap,
        ),
        documents=new_documents,
    )
    stats = BuildStats(
        documents=len(new_documents),
        chunks=total_chunks,
        reused=reused_chunks,
        embedded=embedded_chunks,
    )
    return index, stats
