"""Application settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    agent_model: str
    embedding_model: str
    embedding_dimensions: int
    corpus_dir: Path
    index_path: Path
    expected_documents: int
    chunk_chars: int
    chunk_overlap: int
    top_k: int
    min_similarity: float
    max_context_chars: int
    repo_root: Path


def _parse_int(name: str, value: str, *, positive: bool = True) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if positive and parsed <= 0:
        raise ValueError(f"{name} must be greater than zero, got {parsed}.")
    if not positive and parsed < 0:
        raise ValueError(f"{name} must be zero or greater, got {parsed}.")
    return parsed


def _parse_float(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be zero or greater, got {parsed}.")
    return parsed


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_settings(env: dict[str, str] | None = None, repo_root: Path | None = None) -> Settings:
    """Load and validate settings from the environment."""

    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    load_dotenv(root / ".env", override=False)

    source = env if env is not None else os.environ
    google_api_key = (source.get("GOOGLE_API_KEY") or "").strip() or None
    agent_model = (source.get("AGENT_MODEL") or "gemini-flash-latest").strip()
    embedding_model = (source.get("EMBEDDING_MODEL") or "gemini-embedding-001").strip()
    embedding_dimensions = _parse_int(
        "EMBEDDING_DIMENSIONS", source.get("EMBEDDING_DIMENSIONS", "768")
    )
    expected_documents = _parse_int(
        "EXPECTED_DOCUMENTS", source.get("EXPECTED_DOCUMENTS", "3")
    )
    chunk_chars = _parse_int("CHUNK_CHARS", source.get("CHUNK_CHARS", "1000"))
    chunk_overlap = _parse_int(
        "CHUNK_OVERLAP", source.get("CHUNK_OVERLAP", "150"), positive=False
    )
    top_k = _parse_int("TOP_K", source.get("TOP_K", "5"))
    min_similarity = _parse_float(
        "MIN_SIMILARITY", source.get("MIN_SIMILARITY", "0.45")
    )
    max_context_chars = _parse_int(
        "MAX_CONTEXT_CHARS", source.get("MAX_CONTEXT_CHARS", "6000")
    )
    corpus_dir = _resolve_path(root, source.get("CORPUS_DIR", "docs/corpus"))
    index_path = _resolve_path(root, source.get("INDEX_PATH", "data/index.json"))

    if expected_documents != 3:
        raise ValueError(
            f"EXPECTED_DOCUMENTS must be 3 for this project, got {expected_documents}."
        )
    if chunk_overlap >= chunk_chars:
        raise ValueError(
            f"CHUNK_OVERLAP ({chunk_overlap}) must be less than CHUNK_CHARS ({chunk_chars})."
        )
    if embedding_dimensions not in {128, 256, 512, 768, 1536, 3072}:
        raise ValueError(
            "EMBEDDING_DIMENSIONS must be one of 128, 256, 512, 768, 1536, or 3072, "
            f"got {embedding_dimensions}."
        )
    if not agent_model:
        raise ValueError("AGENT_MODEL must not be empty.")
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL must not be empty.")

    return Settings(
        google_api_key=google_api_key,
        agent_model=agent_model,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        corpus_dir=corpus_dir,
        index_path=index_path,
        expected_documents=expected_documents,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        min_similarity=min_similarity,
        max_context_chars=max_context_chars,
        repo_root=root,
    )


def require_api_key(settings: Settings) -> str:
    if not settings.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to .env in the repository root before running "
            "ingestion or live questions."
        )
    return settings.google_api_key
