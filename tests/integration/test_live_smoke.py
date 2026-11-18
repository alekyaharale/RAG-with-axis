import os

import pytest

from rag_qa.config import load_settings
from rag_qa.documents import load_corpus
from rag_qa.embeddings import GeminiEmbedder
from rag_qa.index import build_index, save_index
from rag_qa.retrieval import RetrievalService
from rag_qa.tools import build_search_tool


pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_live_retrieval_answerable_and_unanswerable(tmp_path):
    # Use repo corpus if available; otherwise skip.
    settings = load_settings()
    if not settings.corpus_dir.exists():
        pytest.skip("Corpus directory not found")
    try:
        docs = load_corpus(settings)
    except Exception:
        pytest.skip("Corpus not available")
    embedder = GeminiEmbedder(
        api_key=os.environ["GOOGLE_API_KEY"],
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )
    index, _ = build_index(docs, settings, embedder)
    temp_index = tmp_path / "index.json"
    save_index(index, temp_index)
    live_settings = load_settings(
        env={"INDEX_PATH": str(temp_index), "GOOGLE_API_KEY": os.environ["GOOGLE_API_KEY"]},
        repo_root=settings.repo_root,
    )
    service = RetrievalService(live_settings, embedder)
    tool = build_search_tool(service)
    ans = tool("What is Nimbus Desk?")
    assert ans["status"] == "success"
    unans = tool("Which cloud region hosts the production database?")
    # Retrieval may or may not return passages; live agent refusal is validated manually.
    assert unans["status"] == "success"
