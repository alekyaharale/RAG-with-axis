# RAG Q&A with Google ADK

A Python command line agent and local ADK app that answer questions using only the three documents in `docs/corpus/`. Every answer must cite the source document and heading, and the agent refuses questions that the corpus does not cover.

## What it is

- `rag_qa/`: ingestion, chunking, embeddings, local JSON index, hybrid retrieval, ADK tool and agent, CLI, and evaluation.
- `docs/corpus/`: three sample PDF documents used for end-to-end runs. Replace them with the real three reference documents when available. Keep exactly three supported files in this directory.
- `eval/questions.json`: sample retrieval questions for `rag-eval`.

The retrieval path is dense cosine similarity over `gemini-embedding-001` embeddings plus BM25, fused with reciprocal rank fusion.

## Data handling

Document text is sent to the Gemini embedding API during ingestion and query embedding. Do not ingest sensitive material into an environment where that data flow is not acceptable. The Gemini API key is read from `GOOGLE_API_KEY` in `.env`.

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- A Gemini API key for ingestion and live answers

## Setup

```bash
uv sync
cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`. `.env` is ignored by git and `.env.example` contains placeholders only.

## Run

Ingest the corpus:

```bash
uv run rag-ingest
```

Ask questions in the CLI:

```bash
uv run rag-qa
```

Use the ADK development interfaces:

```bash
uv run adk run rag_qa
uv run adk web
```

`scripts/dev.sh` runs the one-command path: sync dependencies, create `.env` if missing, stop with setup instructions when the API key is empty, build the index when missing, and start the CLI.

## Test

```bash
uv run pytest
```

Unit and scripted integration tests run without network access. Live smoke tests are marked `live` and require `GOOGLE_API_KEY`:

```bash
uv run pytest -m live
```

Lint and format checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

## Retrieval evaluation

`rag-eval` requires a built index and a live embedding key:

```bash
uv run rag-eval
```

The command reports `recall@5`, `precision@5`, `MRR`, and the unanswerable false-positive rate on `eval/questions.json`. It exits nonzero when recall is below `--min-recall` (default `0.8`).

Measured figures on the bundled sample corpus: not recorded in this checkout because evaluation requires a live `GOOGLE_API_KEY`.

## Corpus and index rules

- `docs/corpus/` holds exactly three PDF documents. Ingestion still accepts `.md`, `.txt`, and `.pdf` files.
- Ingestion fails with a clear error for any other count.
- PDF files must contain an extractable text layer. OCR is not supported.
- The sample PDFs keep Markdown `#` heading markers in their text layer so heading-aware chunking and citations work.
- `data/index.json` stores chunk provenance: source, heading, page, character offsets, content hash, and vectors.

## Boundaries

This project does not include production deployment, authentication, multi-user isolation, persistent sessions, Vertex AI RAG Engine, hosted vector databases, cross-encoder reranking, OCR, multiple corpora, automatic reindexing on file change, or a custom web UI beyond ADK's development interface.

## Contributing

1. Keep changes focused and run `uv run pytest`, `uv run ruff check .`, and `uv run ruff format --check .`.
2. Update tests when behavior changes.
3. Do not commit `.env`, `data/index.json`, or `.venv/`.
