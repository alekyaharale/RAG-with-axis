"""Corpus ingestion command."""

from __future__ import annotations

import sys

from .config import load_settings, require_api_key
from .documents import load_corpus
from .embeddings import EmbeddingError, GeminiEmbedder
from .index import BuildStats, build_index, load_index, save_index, IndexError as StoredIndexError


def ingest() -> BuildStats:
    settings = load_settings()
    api_key = require_api_key(settings)
    documents = load_corpus(settings)
    embedder = GeminiEmbedder(
        api_key=api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    existing = None
    if settings.index_path.exists():
        try:
            existing = load_index(settings.index_path)
        except StoredIndexError:
            existing = None
    index, stats = build_index(documents, settings, embedder, existing=existing)
    save_index(index, settings.index_path)
    return stats


def main() -> None:
    try:
        settings = load_settings()
        stats = ingest()
        print(
            f"{stats.documents} documents, {stats.chunks} chunks, {stats.reused} reused, "
            f"{stats.embedded} embedded -> {settings.index_path}"
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except EmbeddingError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
