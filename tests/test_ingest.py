from pathlib import Path

import pytest

from rag_qa import ingest as ingest_mod


class FakeEmbedder:
    def __init__(self, *args, **kwargs):
        self.dimensions = kwargs.get("dimensions", 768)
    def embed_documents(self, texts):
        return [[0.0] * self.dimensions for _ in texts]
    def embed_query(self, text):
        return [0.0] * self.dimensions


def write_corpus(dir_path: Path, n: int):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (dir_path / f"{i}.md").write_text(f"# Doc {i}\n\ncontent {i}", encoding="utf-8")


def test_ingest_success(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "docs/corpus"
    write_corpus(corpus, 3)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "data/index.json"))
    monkeypatch.setattr(ingest_mod, "GeminiEmbedder", FakeEmbedder)
    monkeypatch.chdir(tmp_path)
    ingest_mod.main()
    out = capsys.readouterr().out
    assert "3 documents" in out
    assert (tmp_path / "data/index.json").exists()


def test_ingest_wrong_count(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "docs/corpus"
    write_corpus(corpus, 2)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "data/index.json"))
    monkeypatch.setattr(ingest_mod, "GeminiEmbedder", FakeEmbedder)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        ingest_mod.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "Found 2" in err
    assert str(corpus) in err


def test_ingest_missing_key(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "docs/corpus"
    write_corpus(corpus, 3)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "data/index.json"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        ingest_mod.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "GOOGLE_API_KEY" in err
