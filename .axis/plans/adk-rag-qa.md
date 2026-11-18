# Plan: Google ADK Q&A agent over three reference documents

**Slug:** `adk-rag-qa` • **Created:** 2026-11-18 • **Base:** `3314f2d`

## Objective

A Python CLI and ADK web app answer natural-language questions using only the three documents in `docs/corpus/`, cite the source document and heading for every answer, and refuse when the documents do not cover the question.

## Context

- **Repo state (observed):** only AXIS scaffolding exists at commit `3314f2d`. There is no application code, dependency manifest, README, or reference document. Python 3.12 and `uv` 0.13 are installed; `make` and the `adk` CLI are not on `PATH`.
- **Reference documents (user-provided gap):** the three real documents are not in the repo. The plan ships three short sample documents so the pipeline runs end to end. Replacing them with the real three requires no code change.
- **Files involved:**
  - Existing: `.gitignore:1-16` (append runtime ignores), `AGENTS.md` (tool and commit conventions), `.axis/playbooks/agentic-development.md`, `.axis/playbooks/app-testing.md`, `.axis/playbooks/core-coding-standards.md`.
  - New package: `rag_qa/{config,documents,chunking,embeddings,index,retrieval,tools,agent,cli,ingest,evaluate}.py`.
  - New tests: `tests/` with unit tests and one scripted-model integration test.
  - New corpus: `docs/corpus/01-*.md`, `02-*.md`, `03-*.md`.
  - New project files: `pyproject.toml`, `.python-version`, `.env.example`, `.editorconfig`, `.pre-commit-config.yaml`, `scripts/dev.sh`, `README.md`.
- **Constraints:**
  - Python 3.12 via `uv`; `google-adk` 2.14.x and `google-genai` 2.x (ADK 2.14 requires `google-genai<3`).
  - Gemini API key comes from `GOOGLE_API_KEY` in `.env`; no other cloud project or hosted vector store.
  - Exactly three documents. Ingestion fails loudly on 0, 2, 4, or more.
  - Supported inputs: Markdown, plain text, and text-layer PDF. OCR is out of scope.
  - Documents are sent to the Gemini embedding API. The README states this data-handling fact.
- **Dependencies:** none upstream. `uv` must be able to resolve the Python 3.12 interpreter and packages from PyPI.

## Approach

**Chosen:** one `LlmAgent` with a single typed retrieval tool, backed by a local JSON index. Ingestion loads the three documents, chunks them on headings and paragraphs, embeds each chunk with `gemini-embedding-001` at 768 dimensions, and writes `data/index.json` with source, heading, page, character offsets, and content hash on every chunk. A question is embedded with `RETRIEVAL_QUERY`; the retriever ranks chunks by dense cosine similarity and BM25, fuses the two ranked lists with reciprocal rank fusion, and returns the top five passages inside a character budget. The agent answers only from those passages and cites them.

**Alternative considered:** ADK's built-in `FilesRetrieval` tool. It builds a LlamaIndex index inside its constructor, re-indexes and re-embeds on every module import, defaults to a preview embedding model, and returns a single chunk without a stable chunk identifier or offsets. That makes incremental ingest, citation tests, and cost control impossible. `VertexAiRagRetrieval` was also rejected because it requires a pre-built Vertex AI RAG corpus and a Google Cloud project, while this task runs with a Gemini API key alone.

**Reasoning:** the boundaries sit where the next changes will land. Replacing the corpus touches `documents.py` and the ingest command. Changing chunking touches `chunking.py` and one config value. Swapping the store for pgvector or Qdrant touches `index.py` and `retrieval.py`. The ADK integration stays a thin agent plus tool layer, which keeps the framework upgrade surface small.

**Left unbuilt:** a cross-encoder reranker, a hosted vector database, multi-corpus or multi-tenant routing, conversation memory that outlives the process, OCR for scanned PDFs, and automatic reindex on file change. A reranker becomes worth building when the eval set shows recall@5 below target after fusion or the corpus grows past roughly one hundred documents. A vector database becomes worth building when the index no longer fits in memory or several writers update it concurrently. OCR becomes necessary when a source PDF has no text layer.

## Steps

1. `[new file: pyproject.toml]` Create the `uv` project for Python 3.12 with runtime dependencies `google-adk>=2.14,<3`, `google-genai>=2.30,<3`, `pypdf>=7,<8`, and `python-dotenv>=1,<2`; add `pytest`, `pytest-asyncio`, `ruff`, and `pre-commit` as dev dependencies; define the `rag-ingest`, `rag-qa`, and `rag-eval` console scripts; set pytest `asyncio_mode = "auto"` and the `live` marker. Test impact: `uv sync` resolves and `uv run pytest --collect-only` starts.
2. `[new file: .python-version]` Pin `3.12`. `[new file: .env.example]` Document every setting with placeholders only: `GOOGLE_API_KEY=`, `AGENT_MODEL=gemini-flash-latest`, `EMBEDDING_MODEL=gemini-embedding-001`, `EMBEDDING_DIMENSIONS=768`, `CORPUS_DIR=docs/corpus`, `INDEX_PATH=data/index.json`, `EXPECTED_DOCUMENTS=3`, `CHUNK_CHARS=1000`, `CHUNK_OVERLAP=150`, `TOP_K=5`, `MIN_SIMILARITY=0.45`, `MAX_CONTEXT_CHARS=6000`. `[.gitignore:1-16]` Append `.env`, `.venv/`, `data/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, and `*.pyc`. Test impact: no source behavior; a secret scan finds no values.
3. `[new file: docs/corpus/01-product-overview.md]` `[new file: docs/corpus/02-support-policy.md]` `[new file: docs/corpus/03-operations-guide.md]` Write three one-page sample documents with three to five headings each, concrete facts, and at least one question that needs facts from two documents. Mark each file's first line as sample content to replace. Test impact: provides a manual end-to-end corpus; unit tests use their own fixtures.
4. `[new file: rag_qa/config.py]` Add a frozen `Settings` dataclass and `load_settings()`, which calls `load_dotenv()` and parses the environment. Validate `EXPECTED_DOCUMENTS == 3`, positive numeric knobs, `CHUNK_OVERLAP < CHUNK_CHARS`, supported embedding dimensions, and resolve `CORPUS_DIR` and `INDEX_PATH` against the repository root. Test impact: `tests/test_config.py` covers defaults, overrides, invalid counts, and invalid numeric values.
5. `[new file: rag_qa/documents.py]` Add frozen `Page` and `SourceDocument` records and `load_corpus()`. Accept only immediate `.md`, `.txt`, and `.pdf` files, sort by filename, hash the normalized text with SHA-256, and reject any file count other than three with a message naming the count, the directory, and the expected count. Extract PDF pages with `pypdf`; an empty text layer fails with the file path and states that OCR is not supported. Test impact: `tests/test_documents.py` covers three files, wrong counts, an unsupported extension, an empty PDF, and hash stability.
6. `[new file: rag_qa/chunking.py]` Add frozen `Chunk` records and `chunk_document()`. Split Markdown on headings while ignoring `#` lines inside fenced code blocks; pack paragraphs up to `CHUNK_CHARS`; split oversized paragraphs on sentence boundaries; overlap by trailing complete sentences up to `CHUNK_OVERLAP`; carry the heading path, PDF page, start and end character offsets, and a content hash. Chunk IDs are stable per document and ordinal (`<source>#<ordinal>`). Test impact: `tests/test_chunking.py` covers heading paths, code fences, overlap, offset reconstruction, PDF pages, short documents, and Unicode.
7. `[new file: rag_qa/embeddings.py]` Add an `EmbeddingClient` protocol with `embed_documents()` and `embed_query()`, plus a `GeminiEmbedder` implementation. Batch document embeddings in groups of 32, use `types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=...)` for documents and `RETRIEVAL_QUERY` for questions, validate every vector's length, and retry 429 and 5xx responses three times with exponential backoff and jitter. Raise `EmbeddingError` naming the operation and the next action; never include the API key. Test impact: `tests/test_embeddings.py` uses a fake client to assert task types, batching, retry, dimension mismatch, and redaction.
8. `[new file: rag_qa/index.py]` Add the index manifest, `build_index()`, `load_index()`, and atomic save. The manifest records schema version, embedding model, dimensions, chunk size, overlap, and per-document SHA-256. Reuse chunks and vectors for a document whose hash and embedding settings match; re-embed only a changed document; drop documents no longer in the corpus; rebuild everything when model, dimensions, or chunk settings change. Write to a temporary file and replace it with `os.replace`, retrying once on `PermissionError` for Windows and OneDrive locks. Test impact: `tests/test_index.py` proves a second build makes zero embedding calls, a changed document re-embeds only itself, a removed document is dropped, a settings mismatch rebuilds, and corrupt JSON fails with a clear error.
9. `[new file: rag_qa/retrieval.py]` Add pure tokenization, BM25 scoring, cosine similarity, reciprocal rank fusion, and a `Retriever.search()` that takes `top_k`, `MIN_SIMILARITY`, and `MAX_CONTEXT_CHARS`. Retrieve twenty dense and twenty lexical candidates, deduplicate by chunk ID, discard dense candidates below the similarity floor, fuse with RRF at k=60, then take the top passages until the character budget is reached. Break ties by document name and ordinal. `RetrievalService` loads the index lazily and raises `IndexNotBuilt` with the exact next command. Test impact: `tests/test_retrieval.py` covers semantic ranking, exact-identifier lexical ranking, threshold filtering, the character cap, deterministic ordering, and an empty result.
10. `[new file: rag_qa/tools.py]` Add `build_search_tool(service)`, returning the function tool `search_reference_documents(query: str) -> dict`. The docstring states the purpose, the `query` argument, and the return shape, because ADK sends the whole docstring to the model. Return `{"status": "success", "query": ..., "count": ..., "results": [...]}` with `source`, `heading`, `page`, `chunk_id`, `score`, and `text` per result. Return `{"status": "error", "error_message": ...}` when the index is missing. Test impact: `tests/test_tools.py` covers the schema, deterministic ordering, JSON serializability, and the error message.
11. `[new file: rag_qa/agent.py]` Add `build_agent(service, settings)` and the module-level `root_agent` and `app = App(name="rag_qa", root_agent=root_agent)`. The instruction uses fixed sections: Role, Context, Task, Format, Constraints. It requires a retrieval call before every answer, limits claims to returned passages, requires citations in the form `[source: <file>, "<heading>"]`, refuses when no passage covers the question, forbids outside knowledge, and treats passage text as untrusted data whose instructions are ignored. Test impact: `tests/test_agent.py` asserts the name, model, single tool, and the citation, refusal, and untrusted-data rules in the instruction.
12. `[new file: rag_qa/cli.py]` Add an async REPL using `InMemoryRunner(app=app)`. Preflight the API key and index, create one session, stream events until the final response, print only the final text, and handle `/exit`, `EOFError`, `KeyboardInterrupt`, and retrieval errors with a next action on stderr. Test impact: `tests/test_cli.py` covers final-text extraction and error formatting; the runner path is covered by the integration test.
13. `[new file: rag_qa/ingest.py]` Add the ingest command: load settings, check the API key, load exactly three documents, build the index, and print `3 documents, N chunks, N reused, N embedded -> data/index.json`. Exit 2 on validation errors and 1 on embedding errors. Test impact: `tests/test_ingest.py` covers success output, wrong document count, and a missing key.
14. `[new file: rag_qa/evaluate.py]` `[new file: eval/questions.json]` Add ten sample-corpus questions, including two unanswerable ones, and a retrieval evaluation that reports recall@5, precision@5, MRR, and the unanswerable false-positive rate. Exit nonzero below `--min-recall 0.8`. The command requires a live embedding key and is not part of the default test run. Test impact: `tests/test_evaluate.py` covers the metric functions with synthetic rankings and validates the question file schema.
15. `[new file: tests/fakes.py]` `[new file: tests/fixtures/corpus/alpha.md]` `[new file: tests/fixtures/corpus/beta.md]` `[new file: tests/fixtures/corpus/gamma.md]` Add a deterministic hash-based `FakeEmbedder` that records calls, and three fixture documents with known facts. Test impact: all unit tests run without network access.
16. `[new file: tests/integration/test_agent_loop.py]` Add a scripted `BaseLlm` that returns a function call for `search_reference_documents`, reads the function response from the next request, and returns a final answer citing the returned source. Run the real `InMemoryRunner` against an in-memory index built with `FakeEmbedder`; assert the tool call, the returned provenance, and the final citation. Add `[new file: tests/integration/test_live_smoke.py]` with one answerable and one unanswerable question under the `live` marker, skipped unless `GOOGLE_API_KEY` is set. Test impact: `uv run pytest` proves the ADK loop without a key; `uv run pytest -m live` proves the real model path.
17. `[new file: scripts/dev.sh]` Add the one-command path: `uv sync`, copy `.env.example` to `.env` when absent, stop with the exact key instruction when `GOOGLE_API_KEY` is empty, run `rag-ingest` when the index is missing, then run `rag-qa`. Use `set -euo pipefail`. Test impact: manual shell run from a clean checkout.
18. `[new file: README.md]` `[new file: .editorconfig]` `[new file: .pre-commit-config.yaml]` Write a one-screen README answering what it is, how to run it, how to test it, and how to contribute. Document the sample corpus, data handling, the three-document rule, the retrieval baseline command, and the boundaries. Add editor settings and pre-commit hooks for gitleaks and ruff. Test impact: no runtime behavior.

## Acceptance criteria

- [ ] `uv sync` resolves on Python 3.12, and `uv run pytest` passes without network access.
- [ ] `uv run rag-ingest` over exactly three documents writes `data/index.json` with a manifest and every chunk's source, heading, page, offsets, and content hash.
- [ ] A second ingest with unchanged documents makes zero embedding calls, proven by a test.
- [ ] A corpus with 0, 2, or 4 documents exits nonzero and names the found count, the directory, and the expected count.
- [ ] `uv run rag-qa` answers a question whose answer is in one document and includes a citation naming that document and heading.
- [ ] A question absent from the corpus produces a refusal that names the reference documents, not an invented answer.
- [ ] `uv run adk run rag_qa` starts, and `uv run adk web` lists the `rag_qa` agent.
- [ ] `uv run rag-eval` reports recall@5 at or above 0.8 on the sample question set, and the README records the measured figures.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [ ] No secret value is committed; `.env` is ignored and `.env.example` contains placeholders only.
- [ ] All existing tests still pass. The repository has none today.

## Risks & rollback

- **Risk:** A missing or invalid Gemini key blocks ingestion and live answers. **Mitigation:** preflight both commands, name `GOOGLE_API_KEY` and the `.env` location, and never print the value.
- **Risk:** The embedding model or dimensions change and old vectors become incomparable. **Mitigation:** store model and dimensions in the manifest; a mismatch triggers a full rebuild with a printed notice.
- **Risk:** A PDF has no text layer, so chunks are empty. **Mitigation:** fail with the file path and state that OCR is unsupported.
- **Risk:** The model answers from outside knowledge or ignores the tool. **Mitigation:** the instruction requires retrieval and citations, the tool is the only source of document text, and the scripted integration test proves the tool loop; a live smoke test checks the refusal path.
- **Risk:** A reference document contains prompt injection. **Mitigation:** pass passages as structured tool results, instruct the model that passage text is data and its instructions are ignored, and keep document text out of the system instruction.
- **Risk:** Retrieved passages exceed the model's useful context budget. **Mitigation:** top five passages and a 6000-character cap, both configurable.
- **Risk:** The index becomes expensive to rebuild as the corpus grows. **Mitigation:** document-level content hashes reuse unchanged vectors; the hosted vector store is listed as the next boundary.
- **Risk:** `os.replace` fails on a locked OneDrive file. **Mitigation:** write in the same directory and retry once after a short delay.
- **Risk:** Documents contain sensitive material sent to the Gemini API. **Mitigation:** README states the data flow before the first ingest; sensitive corpora require a different deployment, which is out of scope.
- **Risk:** ADK 2.x changes a public import or runner signature. **Mitigation:** pin `google-adk<3`, import only documented public classes, and cover the agent loop with an integration test.
- **Rollback:** remove `rag_qa/`, `tests/`, `docs/corpus/`, `eval/`, `scripts/`, `data/index.json`, and `.venv/`; revert the additions to `.gitignore` and delete the new project files. No deployed service, database migration, or persisted user data exists.

## Out of scope

- Production deployment, authentication, multi-user isolation, and persistent sessions.
- Vertex AI RAG Engine, Gemini File Search, or a hosted vector database.
- A cross-encoder reranker or query rewriting.
- OCR for scanned PDFs and non-text formats such as DOCX or HTML.
- Multiple corpora, tenant routing, or per-user document access.
- An LLM-judge faithfulness metric. The eval covers retrieval quality; answer grounding is checked by the scripted test and the live smoke questions.
- Automatic reindex on file change. Ingest runs explicitly.
- A custom web UI beyond ADK's development web interface.

---

## Execution log — 2026-11-18 10:10

**Blocked at step verification:** `AGENTS.md` does not define a project test command in a `## Test` section, so the required end-of-run test command could not be executed without guessing.
**State on exit:** All planned files for `adk-rag-qa` were created and staged, including package code, tests, sample corpus, eval assets, scripts, and project configuration. No tests were run.
**Suggested next:** Add a `## Test` section to `AGENTS.md` with the project test command (for example `uv sync` then `uv run pytest`), run it, and review `git diff --cached` before committing. If the command passes, mark the backlog entry done and continue with `/axis-distill` after the commit lands on main.
