#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv sync

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example."
fi

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a
  . ./.env
  set +a
fi

if [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "GOOGLE_API_KEY is empty. Set it in .env in the repository root, then run scripts/dev.sh again." >&2
  exit 2
fi

if [ ! -f "${INDEX_PATH:-data/index.json}" ]; then
  uv run rag-ingest
fi

uv run rag-qa
