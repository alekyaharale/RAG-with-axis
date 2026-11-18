from rag_qa.config import load_settings
from rag_qa.documents import SourceDocument, Page
from rag_qa.index import build_index
from rag_qa.retrieval import Retriever, RetrievalService, IndexNotBuilt
from tests.fakes import FakeEmbedder


def make_doc(name: str, text: str) -> SourceDocument:
    import hashlib
    return SourceDocument(
        name=name,
        path=None,  # type: ignore
        pages=(Page(None, text),),
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def build_test_index(tmp_path):
    settings = load_settings(env={}, repo_root=tmp_path)
    docs = [
        make_doc("a.md", "# Alpha\n\nOnboarding uses workspace identifiers like WS-12345."),
        make_doc("b.md", "# Beta\n\nAudit logs are retained for 30 days and reviewed quarterly."),
        make_doc("c.md", "# Gamma\n\nMaintenance occurs Sundays at 02:00 with weekly backups."),
    ]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index, _ = build_index(docs, settings, embedder)
    return index, settings, embedder


def test_retriever_semantic_and_lexical(tmp_path):
    index, settings, embedder = build_test_index(tmp_path)
    retriever = Retriever(index)
    # lexical/exact identifier
    q = "Where is WS-12345 used?"
    qv = embedder.embed_query(q)
    results = retriever.search(q, qv, top_k=5, min_similarity=0.0, max_context_chars=6000)
    assert results
    assert any("WS-12345" in r.chunk.text for r in results)
    # semantic-ish by token overlap
    q2 = "How long are audit logs kept?"
    qv2 = embedder.embed_query(q2)
    results2 = retriever.search(q2, qv2, top_k=5, min_similarity=0.0, max_context_chars=6000)
    assert results2
    assert results2[0].chunk.source == "b.md"


def test_retriever_threshold_and_cap_and_determinism(tmp_path):
    index, settings, embedder = build_test_index(tmp_path)
    retriever = Retriever(index)
    q = "completely unrelated zebra xylophone"
    qv = [0.0] * settings.embedding_dimensions
    results = retriever.search(q, qv, top_k=5, min_similarity=0.99, max_context_chars=6000)
    assert results == []
    # cap
    q2 = "audit maintenance onboarding"
    qv2 = embedder.embed_query(q2)
    capped = retriever.search(q2, qv2, top_k=5, min_similarity=0.0, max_context_chars=40)
    assert len(capped) <= 5
    if len(capped) > 1:
        assert sum(len(r.chunk.text) for r in capped) <= 40 + len(capped[-1].chunk.text)
    r1 = retriever.search(q2, qv2, top_k=5, min_similarity=0.0, max_context_chars=6000)
    r2 = retriever.search(q2, qv2, top_k=5, min_similarity=0.0, max_context_chars=6000)
    assert [r.chunk.chunk_id for r in r1] == [r.chunk.chunk_id for r in r2]


def test_retrieval_service_missing_index(tmp_path):
    settings = load_settings(env={"INDEX_PATH": str(tmp_path / "missing.json")}, repo_root=tmp_path)
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    service = RetrievalService(settings, embedder)
    try:
        service.search("hello")
        assert False
    except IndexNotBuilt as exc:
        assert "rag-ingest" in str(exc)
