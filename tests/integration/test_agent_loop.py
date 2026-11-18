import asyncio

import pytest

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from rag_qa.agent import INSTRUCTION
from rag_qa.config import load_settings
from rag_qa.documents import Page, SourceDocument
from rag_qa.index import build_index, save_index
from rag_qa.retrieval import RetrievalService
from rag_qa.tools import build_search_tool
from tests.fakes import FakeEmbedder


def make_doc(name: str, text: str) -> SourceDocument:
    import hashlib

    return SourceDocument(
        name=name,
        path=None,  # type: ignore[arg-type]
        pages=(Page(None, text),),
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


class ScriptedLlm(BaseLlm):
    def __init__(self):
        super().__init__(model="scripted")
        self.calls = 0

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        # If the last user/tool message contains function response, answer with citation.
        contents = getattr(llm_request, "contents", []) or []
        has_function_response = False
        source = "a.md"
        heading = "Alpha"
        for content in contents:
            for part in getattr(content, "parts", []) or []:
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    has_function_response = True
                    resp = getattr(fr, "response", None) or {}
                    results = resp.get("results") or []
                    if results:
                        source = results[0].get("source", source)
                        heading = results[0].get("heading", heading)
        if not has_function_response:
            part = types.Part.from_function_call(
                name="search_reference_documents", args={"query": "alpha fact"}
            )
            content = types.Content(role="model", parts=[part])
        else:
            text = f"Alpha fact is available [source: {source}, \"{heading}\"]."
            content = types.Content(role="model", parts=[types.Part(text=text)])

        response = types.GenerateContentResponse(
            candidates=[types.Candidate(content=content, finish_reason="STOP")]
        )
        yield types.LlmResponse(
            content=content, raw_response=response, turn_complete=True
        )

    def connect(self, llm_request):  # pragma: no cover - not used
        raise NotImplementedError


@pytest.mark.asyncio
async def test_agent_loop_with_scripted_llm(tmp_path):
    settings = load_settings(env={"INDEX_PATH": str(tmp_path / "index.json")}, repo_root=tmp_path)
    docs = [
        make_doc("a.md", "# Alpha\n\nAlpha fact is here."),
        make_doc("b.md", "# Beta\n\nBeta fact."),
        make_doc("c.md", "# Gamma\n\nGamma fact."),
    ]
    embedder = FakeEmbedder(dimensions=settings.embedding_dimensions)
    index, _ = build_index(docs, settings, embedder)
    save_index(index, settings.index_path)
    service = RetrievalService(settings, embedder)
    tool = build_search_tool(service)

    llm = ScriptedLlm()
    agent = LlmAgent(
        name="rag_qa",
        model=llm,
        instruction=INSTRUCTION,
        tools=[tool],
    )
    # Build an App wrapper for runner
    from google.adk.apps import App

    app = App(name="rag_qa", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name=app.name, user_id="u")
    message = types.Content(role="user", parts=[types.Part(text="What is the alpha fact?")])
    events = []
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=message
    ):
        events.append(event)

    # Find function response and final text
    saw_tool_response = False
    final_text = ""
    for event in events:
        content = getattr(event, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            fr = getattr(part, "function_response", None)
            if fr is not None:
                saw_tool_response = True
                resp = getattr(fr, "response", {}) or {}
                assert resp.get("status") in {"success", "error"}
                if resp.get("status") == "success":
                    assert resp.get("results")
                    assert resp["results"][0].get("source")
            if getattr(part, "text", None) and getattr(event, "author", "") != "user":
                final_text = part.text
    assert llm.calls >= 2
    assert saw_tool_response
    assert "[source:" in final_text
