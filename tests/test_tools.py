import json

from rag_qa.config import load_settings
from rag_qa.documents import SourceDocument, Page
from rag_qa.index import build_index, save_index
from rag_qa.retrieval import RetrievalService
from rag_qa.tools import build_search_tool
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


def test_search_tool_success_and_schema(tmp_path):
    settings = load_settings(env={"INDEX_PATH": str(tmp_path / "index.json")}, repo_root=tmp_path)
    docs = [
        make_doc("a.md", "# A\n\nAlpha fact."),
        make_doc("b.md", "# B\n\nBeta fact."),
        make_doc("c.md", "# C\n\nGamma fact."),
    ]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index, _ = build_index(docs, settings, embedder)
    save_index(index, settings.index_path)
    service = RetrievalService(settings, embedder)
    tool = build_search_tool(service)
    out = tool("Alpha fact")
    assert out["status"] == "success"
    assert "results" in out and isinstance(out["results"], list)
    assert out["count"] == len(out["results"])
    if out["results"]:
        r = out["results"][0]
        for key in ["source", "heading", "page", "chunk_id", "score", "text"]:
            assert key in r
    json.dumps(out)  # serializable
    out2 = tool("Alpha fact")
    assert [r["chunk_id"] for r in out["results"]] == [r["chunk_id"] for r in out2["results"]]


def test_search_tool_error(tmp_path):
    settings = load_settings(env={"INDEX_PATH": str(tmp_path / "missing.json")}, repo_root=tmp_path)
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    service = RetrievalService(settings, embedder)
    tool = build_search_tool(service)
    out = tool("question")
    assert out["status"] == "error"
    assert "rag-ingest" in out["error_message"]
    json.dumps(out)
