"""ADK tool definitions for retrieval."""

from __future__ import annotations

from typing import Any, Callable

from .retrieval import IndexNotBuilt, RetrievalService


def build_search_tool(service: RetrievalService) -> Callable[[str], dict[str, Any]]:
    def search_reference_documents(query: str) -> dict[str, Any]:
        """Search the approved reference documents for passages relevant to a question.

        Args:
            query: The user's question or a focused search phrase.

        Returns:
            A dictionary with status "success" and a "results" list of passages, or status
            "error" with an "error_message". Each result includes source, heading, page,
            chunk_id, score, and text.
        """

        try:
            results = service.search(query)
        except IndexNotBuilt as exc:
            return {"status": "error", "error_message": str(exc)}
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return {
                "status": "error",
                "error_message": f"Search failed: {exc}. Check the index and try again.",
            }

        return {
            "status": "success",
            "query": query,
            "count": len(results),
            "results": [result.to_payload() for result in results],
        }

    return search_reference_documents
