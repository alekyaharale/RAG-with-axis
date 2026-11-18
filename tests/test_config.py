from pathlib import Path

import pytest

from rag_qa.config import load_settings


def test_load_settings_defaults(tmp_path: Path):
    settings = load_settings(env={}, repo_root=tmp_path)
    assert settings.expected_documents == 3
    assert settings.chunk_chars == 1000
    assert settings.chunk_overlap == 150
    assert settings.top_k == 5
    assert settings.min_similarity == 0.45
    assert settings.corpus_dir == (tmp_path / "docs/corpus").resolve()
    assert settings.index_path == (tmp_path / "data/index.json").resolve()


def test_load_settings_overrides(tmp_path: Path):
    env = {
        "GOOGLE_API_KEY": "test-key",
        "AGENT_MODEL": "gemini-test",
        "EMBEDDING_MODEL": "embed-test",
        "EMBEDDING_DIMENSIONS": "256",
        "CORPUS_DIR": "mydocs",
        "INDEX_PATH": "out/idx.json",
        "CHUNK_CHARS": "500",
        "CHUNK_OVERLAP": "50",
        "TOP_K": "3",
        "MIN_SIMILARITY": "0.2",
        "MAX_CONTEXT_CHARS": "2000",
    }
    settings = load_settings(env=env, repo_root=tmp_path)
    assert settings.google_api_key == "test-key"
    assert settings.agent_model == "gemini-test"
    assert settings.embedding_dimensions == 256
    assert settings.corpus_dir == (tmp_path / "mydocs").resolve()
    assert settings.index_path == (tmp_path / "out/idx.json").resolve()


def test_load_settings_rejects_wrong_count(tmp_path: Path):
    with pytest.raises(ValueError, match="EXPECTED_DOCUMENTS must be 3"):
        load_settings(env={"EXPECTED_DOCUMENTS": "2"}, repo_root=tmp_path)


def test_load_settings_rejects_invalid_numbers(tmp_path: Path):
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        load_settings(env={"CHUNK_OVERLAP": "1000"}, repo_root=tmp_path)
    with pytest.raises(ValueError):
        load_settings(env={"CHUNK_CHARS": "0"}, repo_root=tmp_path)
    with pytest.raises(ValueError):
        load_settings(env={"EMBEDDING_DIMENSIONS": "123"}, repo_root=tmp_path)
