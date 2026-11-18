"""Interactive CLI for the RAG agent."""

from __future__ import annotations

import asyncio
import sys

from google.adk.runners import InMemoryRunner
from google.genai import types

from .agent import app
from .config import load_settings, require_api_key
from .index import load_index, IndexError as StoredIndexError


def _extract_final_text(events: list[object]) -> str:
    for event in reversed(events):
        content = getattr(event, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        texts = [getattr(part, "text", "") for part in parts if getattr(part, "text", None)]
        if texts:
            return "\n".join(texts).strip()
    return ""


async def run_repl() -> int:
    settings = load_settings()
    try:
        require_api_key(settings)
        load_index(settings.index_path)
    except (ValueError, StoredIndexError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name=app.name, user_id="local-user"
    )
    print("rag-qa ready. Type /exit to quit.")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0
        if not question:
            continue
        if question == "/exit":
            return 0
        message = types.Content(role="user", parts=[types.Part(text=question)])
        events: list[object] = []
        try:
            async for event in runner.run_async(
                user_id=session.user_id,
                session_id=session.id,
                new_message=message,
            ):
                events.append(event)
            answer = _extract_final_text(events)
            if answer:
                print(answer)
            else:
                print("No answer was produced. Try asking again.", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"Error: {exc}", file=sys.stderr)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run_repl()))


if __name__ == "__main__":
    main()
