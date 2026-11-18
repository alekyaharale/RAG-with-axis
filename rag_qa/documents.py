"""Corpus loading and document extraction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from pypdf import PdfReader

from .config import SUPPORTED_EXTENSIONS, Settings


@dataclass(frozen=True)
class Page:
    number: int | None
    text: str


@dataclass(frozen=True)
class SourceDocument:
    name: str
    path: Path
    pages: tuple[Page, ...]
    text: str
    content_hash: str


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> tuple[Page, ...]:
    reader = PdfReader(str(path))
    pages: list[Page] = []
    any_text = False
    for index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        normalized = _normalize_text(extracted)
        if normalized:
            any_text = True
        pages.append(Page(number=index, text=normalized))
    if not any_text:
        raise ValueError(
            f"PDF '{path}' has no extractable text layer. OCR is not supported; provide a text "
            "PDF or a Markdown/text document."
        )
    return tuple(pages)


def load_document(path: Path) -> SourceDocument:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type '{suffix}' for '{path.name}'. Supported types: "
            ".md, .txt, .pdf."
        )
    if suffix == ".pdf":
        pages = _read_pdf(path)
        text = _normalize_text("\n\n".join(page.text for page in pages if page.text))
    else:
        text = _normalize_text(_read_text_file(path))
        if not text:
            raise ValueError(f"Document '{path}' is empty.")
        pages = (Page(number=None, text=text),)

    content_hash = sha256(text.encode("utf-8")).hexdigest()
    return SourceDocument(
        name=path.name,
        path=path,
        pages=pages,
        text=text,
        content_hash=content_hash,
    )


def load_corpus(settings: Settings) -> list[SourceDocument]:
    corpus_dir = settings.corpus_dir
    if not corpus_dir.exists():
        raise ValueError(
            f"Corpus directory '{corpus_dir}' does not exist. Expected "
            f"{settings.expected_documents} documents."
        )
    files = [
        path
        for path in corpus_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort(key=lambda path: path.name)
    if len(files) != settings.expected_documents:
        raise ValueError(
            f"Found {len(files)} supported document(s) in '{corpus_dir}', expected "
            f"{settings.expected_documents}. Add or remove documents so the corpus contains "
            "exactly three files."
        )
    return [load_document(path) for path in files]
