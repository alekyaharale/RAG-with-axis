import json
from pathlib import Path

from rag_qa.config import load_settings
from rag_qa.documents import SourceDocument, Page
from rag_qa.index import build_index, load_index, save_index
from tests.fakes import FakeEmbedder


def make_doc(name: str, text: str) -> SourceDocument:
    return SourceDocument(
        name=name,
        path=Path(name),
        pages=(Page(None, text),),
        text=text,
        content_hash=__import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
    )


def base_settings(tmp_path: Path, **overrides):
    env = {
        "CORPUS_DIR": str(tmp_path / "corpus"),
        "INDEX_PATH": str(tmp_path / "data/index.json"),
    }
    env.update({k: str(v) for k, v in overrides.items()})
    return load_settings(env=env, repo_root=tmp_path)


def test_build_reuses_unchanged_documents(tmp_path: Path):
    settings = base_settings(tmp_path)
    docs = [make_doc("a.md", "# A\n\none"), make_doc("b.md", "# B\n\ntwo"), make_doc("c.md", "# C\n\nthree")]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index1, stats1 = build_index(docs, settings, embedder)
    assert stats1.embedded > 0
    assert stats1.reused == 0
    calls_after_first = sum(len(c) for c in embedder.document_calls)
    index2, stats2 = build_index(docs, settings, embedder, existing=index1)
    assert stats2.embedded == 0
    assert stats2.reused == stats1.chunks
    assert sum(len(c) for c in embedder.document_calls) == calls_after_first


def test_build_reembeds_only_changed_document(tmp_path: Path):
    settings = base_settings(tmp_path)
    docs = [make_doc("a.md", "# A\n\none"), make_doc("b.md", "# B\n\ntwo"), make_doc("c.md", "# C\n\nthree")]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index1, _ = build_index(docs, settings, embedder)
    docs2 = [docs[0], make_doc("b.md", "# B\n\ntwo changed"), docs[2]]
    before = sum(len(c) for c in embedder.document_calls)
    index2, stats2 = build_index(docs2, settings, embedder, existing=index1)
    assert stats2.embedded > 0
    assert stats2.reused > 0
    assert stats2.embedded < stats2.chunks
    assert sum(len(c) for c in embedder.document_calls) > before
    assert index2.documents["a.md"].content_hash == index1.documents["a.md"].content_hash


def test_build_drops_removed_document(tmp_path: Path):
    settings = base_settings(tmp_path)
    docs = [make_doc("a.md", "# A\n\none"), make_doc("b.md", "# B\n\ntwo"), make_doc("c.md", "# C\n\nthree")]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index1, _ = build_index(docs, settings, embedder)
    docs2 = [docs[0], docs[2], make_doc("d.md", "# D\n\nfour")]
    index2, stats2 = build_index(docs2, settings, embedder, existing=index1)
    assert "b.md" not in index2.documents
    assert "d.md" in index2.documents


def test_settings_mismatch_rebuilds(tmp_path: Path):
    settings = base_settings(tmp_path)
    docs = [make_doc("a.md", "# A\n\none"), make_doc("b.md", "# B\n\ntwo"), make_doc("c.md", "# C\n\nthree")]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index1, _ = build_index(docs, settings, embedder)
    settings2 = base_settings(tmp_path, CHUNK_CHARS=500)
    index2, stats2 = build_index(docs, settings2, embedder, existing=index1)
    assert stats2.reused == 0
    assert stats2.embedded == stats2.chunks


def test_save_load_and_corrupt_json(tmp_path: Path):
    settings = base_settings(tmp_path)
    docs = [make_doc("a.md", "# A\n\none"), make_doc("b.md", "# B\n\ntwo"), make_doc("c.md", "# C\n\nthree")]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index, _ = build_index(docs, settings, embedder)
    path = settings.index_path
    save_index(index, path)
    loaded = load_index(path)
    assert set(loaded.documents.keys()) == {"a.md", "b.md", "c.md"}
    assert loaded.all_chunks()[0].chunk_id
    path.write_text("{bad json", encoding="utf-8")
    try:
        load_index(path)
        assert False, "expected error"
    except Exception as exc:
        assert "not valid JSON" in str(exc) or "unsupported" in str(exc)
