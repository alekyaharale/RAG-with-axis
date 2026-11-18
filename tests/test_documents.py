import re
from pathlib import Path

import pytest

from rag_qa.config import load_settings
from rag_qa.documents import load_corpus, load_document


def make_corpus(dir_path: Path, names: list[str], content: str = "content") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (dir_path / name).write_text(f"# {name}\n\n{content}", encoding="utf-8")


def test_load_corpus_requires_three_documents(tmp_path: Path):
    corpus_dir = tmp_path / "docs/corpus"
    make_corpus(corpus_dir, ["a.md", "b.md", "c.md"])
    settings = load_settings(env={"CORPUS_DIR": str(corpus_dir)}, repo_root=tmp_path)
    docs = load_corpus(settings)
    assert len(docs) == 3
    assert [d.name for d in docs] == ["a.md", "b.md", "c.md"]


@pytest.mark.parametrize("count", [0, 2, 4])
def test_load_corpus_rejects_wrong_count(tmp_path: Path, count: int):
    corpus_dir = tmp_path / "docs/corpus"
    names = [f"doc{i}.md" for i in range(count)]
    if names:
        make_corpus(corpus_dir, names)
    else:
        corpus_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings(env={"CORPUS_DIR": str(corpus_dir)}, repo_root=tmp_path)
    with pytest.raises(ValueError) as exc:
        load_corpus(settings)
    msg = str(exc.value)
    assert f"Found {count}" in msg
    assert str(corpus_dir) in msg
    assert "expected 3" in msg


def test_load_document_rejects_unsupported_extension(tmp_path: Path):
    path = tmp_path / "doc.docx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported document type"):
        load_document(path)


def test_hash_stability(tmp_path: Path):
    path = tmp_path / "a.md"
    path.write_text("# Title\n\nHello world.", encoding="utf-8")
    d1 = load_document(path)
    d2 = load_document(path)
    assert d1.content_hash == d2.content_hash


def test_repo_corpus_is_three_pdf_documents():
    settings = load_settings(env={})
    docs = load_corpus(settings)

    names = [doc.name for doc in docs]
    assert len(names) == 3
    assert all(name.endswith(".pdf") for name in names)
    for doc in docs:
        assert doc.text.strip()
        assert any(re.match(r"^#+ ", line) for line in doc.text.splitlines())
