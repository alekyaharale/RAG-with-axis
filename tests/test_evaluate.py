import json
from pathlib import Path

from rag_qa.config import load_settings
from rag_qa.documents import SourceDocument, Page
from rag_qa.evaluate import evaluate_questions, load_questions, mrr, precision_at_k, recall_at_k
from rag_qa.index import build_index, save_index
from rag_qa.retrieval import RetrievalService
from tests.fakes import FakeEmbedder


def make_doc(name: str, text: str) -> SourceDocument:
    import hashlib
    return SourceDocument(name=name, path=None, pages=(Page(None, text),), text=text, content_hash=hashlib.sha256(text.encode()).hexdigest())  # type: ignore


def test_metric_functions():
    assert recall_at_k(["a.md"], ["a.md", "b.md"], 5) == 1.0
    assert recall_at_k(["a.md"], ["b.md"], 5) == 0.0
    assert precision_at_k(["a.md"], ["a.md", "b.md"], 2) == 0.5
    assert precision_at_k([], [], 5) == 1.0
    assert precision_at_k([], ["a.md"], 5) == 0.0
    assert mrr(["b.md"], ["a.md", "b.md"], ) == 0.5


def test_load_questions_schema(tmp_path):
    path = Path("eval/questions.json")
    questions = load_questions(path)
    assert isinstance(questions, list)
    assert len(questions) == 10
    for q in questions:
        assert "id" in q and "question" in q and "answerable" in q


def test_evaluate_questions_with_fake(tmp_path):
    settings = load_settings(env={"INDEX_PATH": str(tmp_path / "index.json")}, repo_root=tmp_path)
    docs = [
        make_doc("01-product-overview.pdf", "# Nimbus Desk\n\nNimbus Desk is an internal knowledge portal."),
        make_doc("02-support-policy.pdf", "# Support\n\nStandard support is 08:00 to 18:00. Critical incident first response within 15 minutes. Service availability or more than five users escalated."),
        make_doc("03-operations-guide.pdf", "# Operations\n\nPlanned maintenance first Saturday of each month 01:00 to 03:00. Updates every 30 minutes. Verify three reference documents before release."),
    ]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index, _ = build_index(docs, settings, embedder)
    save_index(index, settings.index_path)
    service = RetrievalService(settings, embedder)
    questions = load_questions(Path("eval/questions.json"))
    metrics = evaluate_questions(service, questions, k=5)
    assert "recall_at_k" in metrics and "unanswerable_false_positive_rate" in metrics
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
