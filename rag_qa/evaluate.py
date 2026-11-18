"""Retrieval evaluation for the sample corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_settings, require_api_key
from .embeddings import GeminiEmbedder
from .retrieval import RetrievalService


def load_questions(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Question file must contain a JSON array.")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each question entry must be an object.")
        for key in ("id", "question", "answerable", "expected_sources", "expected_contains"):
            if key not in item:
                raise ValueError(f"Question entry is missing required key '{key}'.")
    return data


def recall_at_k(expected_sources: list[str], retrieved_sources: list[str], k: int) -> float:
    if not expected_sources:
        return 1.0
    retrieved = set(retrieved_sources[:k])
    return 1.0 if any(source in retrieved for source in expected_sources) else 0.0


def precision_at_k(expected_sources: list[str], retrieved_sources: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_sources[:k]
    if not expected_sources:
        return 1.0 if not top else 0.0
    relevant = sum(1 for source in top if source in expected_sources)
    return relevant / len(top) if top else 0.0


def mrr(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    for idx, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return 1.0 / idx
    return 0.0


def evaluate_questions(service: RetrievalService, questions: list[dict[str, object]], k: int = 5):
    answerable = [q for q in questions if q.get("answerable")]
    unanswerable = [q for q in questions if not q.get("answerable")]
    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    false_positives = 0

    for q in answerable:
        results = service.search(str(q["question"]))
        sources = [r.chunk.source for r in results]
        expected = [str(s) for s in q.get("expected_sources", [])]
        recalls.append(recall_at_k(expected, sources, k))
        precisions.append(precision_at_k(expected, sources, k))
        mrrs.append(mrr(expected, sources))

    for q in unanswerable:
        results = service.search(str(q["question"]))
        if results:
            false_positives += 1

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "questions": len(questions),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "recall_at_k": avg(recalls),
        "precision_at_k": avg(precisions),
        "mrr": avg(mrrs),
        "unanswerable_false_positive_rate": (
            false_positives / len(unanswerable) if unanswerable else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    settings = load_settings()
    try:
        api_key = require_api_key(settings)
        questions = load_questions(Path(args.questions))
        embedder = GeminiEmbedder(
            api_key=api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        service = RetrievalService(settings=settings, embedder=embedder)
        metrics = evaluate_questions(service, questions, k=args.k)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(metrics, indent=2))
    if metrics["recall_at_k"] < args.min_recall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
