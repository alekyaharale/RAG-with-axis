from rag_qa.agent import INSTRUCTION, build_agent
from rag_qa.config import load_settings
from rag_qa.documents import SourceDocument, Page
from rag_qa.index import build_index, save_index
from rag_qa.retrieval import RetrievalService
from tests.fakes import FakeEmbedder


def make_doc(name: str, text: str) -> SourceDocument:
    import hashlib
    return SourceDocument(name=name, path=None, pages=(Page(None, text),), text=text, content_hash=hashlib.sha256(text.encode()).hexdigest())  # type: ignore


def test_agent_definition(tmp_path):
    settings = load_settings(env={"INDEX_PATH": str(tmp_path / "index.json")}, repo_root=tmp_path)
    docs = [make_doc("a.md", "# A\n\na"), make_doc("b.md", "# B\n\nb"), make_doc("c.md", "# C\n\nc")]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index, _ = build_index(docs, settings, embedder)
    save_index(index, settings.index_path)
    service = RetrievalService(settings, embedder)
    agent = build_agent(service, settings)
    assert agent.name == "rag_qa"
    assert agent.model == settings.agent_model
    assert len(agent.tools) == 1
    assert "search_reference_documents" in INSTRUCTION or "search_reference_documents" in str(agent.tools[0])
    assert "citation" in INSTRUCTION.lower() or "[source:" in INSTRUCTION
    assert "refuse" in INSTRUCTION.lower()
    assert "untrusted data" in INSTRUCTION.lower() or "ignore any instructions" in INSTRUCTION.lower()
    assert "outside knowledge" in INSTRUCTION.lower()
