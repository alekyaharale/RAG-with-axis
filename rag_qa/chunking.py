"""Document chunking with heading-aware boundaries and provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from .documents import SourceDocument


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    heading: str
    page: int | None
    start_char: int
    end_char: int
    text: str
    content_hash: str
    ordinal: int


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_large_text(text: str, chunk_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    sentences = _sentence_split(text)
    if not sentences:
        return [text[i : i + chunk_chars].strip() for i in range(0, len(text), chunk_chars) if text[i : i + chunk_chars].strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if len(sentence) > chunk_chars:
            if current:
                pieces.append(" ".join(current).strip())
                current = []
                current_len = 0
            for i in range(0, len(sentence), chunk_chars):
                part = sentence[i : i + chunk_chars].strip()
                if part:
                    pieces.append(part)
            continue
        extra = len(sentence) + (1 if current else 0)
        if current and current_len + extra > chunk_chars:
            pieces.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += extra
    if current:
        pieces.append(" ".join(current).strip())
    return [piece for piece in pieces if piece]


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text
    sentences = _sentence_split(text)
    if len(sentences) <= 1:
        tail = text[-overlap:].strip()
        return tail
    selected: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        extra = len(sentence) + (1 if selected else 0)
        if selected and total + extra > overlap:
            break
        selected.insert(0, sentence)
        total += extra
        if total >= overlap:
            break
    joined = " ".join(selected).strip()
    return joined if joined else text[-overlap:].strip()


def _iter_sections(text: str) -> list[tuple[str, str]]:
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[str] = []
    in_code = False
    buffer: list[str] = []
    current_heading = "Document"

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if content:
            sections.append((current_heading, content))
        buffer = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            buffer.append(line)
            continue
        match = _HEADING_RE.match(line)
        if match and not in_code:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading = " > ".join(heading_stack) if heading_stack else title
            continue
        buffer.append(line)
    flush()
    if not sections and text.strip():
        sections.append(("Document", text.strip()))
    # return as (heading, text)
    return [(h, t) for h, t in sections]


def _page_ranges(document: SourceDocument) -> list[tuple[int | None, int, int]]:
    ranges: list[tuple[int | None, int, int]] = []
    search = 0
    for page in document.pages:
        text = page.text
        if not text:
            continue
        idx = document.text.find(text, search)
        if idx == -1:
            idx = document.text.find(text)
        if idx == -1:
            idx = min(search, len(document.text))
        ranges.append((page.number, idx, idx + len(text)))
        search = idx + len(text)
    return ranges


def _page_for_offset(pages: list[tuple[int | None, int, int]], start: int, end: int) -> int | None:
    if not pages:
        return None
    for number, p_start, p_end in pages:
        if start < p_end and end > p_start:
            return number
    return pages[-1][0]


def chunk_document(
    document: SourceDocument, *, chunk_chars: int = 1000, chunk_overlap: int = 150
) -> list[Chunk]:
    if chunk_overlap >= chunk_chars:
        raise ValueError("chunk_overlap must be less than chunk_chars")
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be greater than zero")

    page_ranges = _page_ranges(document)
    chunks: list[Chunk] = []
    ordinal = 0
    search_from = 0

    for heading, section_text in _iter_sections(document.text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        if not paragraphs:
            paragraphs = [section_text.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            units.extend(_split_large_text(paragraph, chunk_chars))

        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            if current and len(candidate) > chunk_chars:
                chunk_text = current.strip()
                idx = document.text.find(chunk_text, search_from)
                if idx == -1:
                    idx = document.text.find(chunk_text)
                if idx == -1:
                    idx = search_from
                start = idx
                end = start + len(chunk_text)
                search_from = start + max(1, len(chunk_text) // 2)
                chunk_hash = sha256(chunk_text.encode("utf-8")).hexdigest()
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.name}#{ordinal:04d}",
                        source=document.name,
                        heading=heading,
                        page=_page_for_offset(page_ranges, start, end),
                        start_char=start,
                        end_char=end,
                        text=chunk_text,
                        content_hash=chunk_hash,
                        ordinal=ordinal,
                    )
                )
                ordinal += 1
                overlap = _tail_overlap(chunk_text, chunk_overlap)
                current = f"{overlap}\n\n{unit}".strip() if overlap else unit
            else:
                current = candidate

        if current.strip():
            chunk_text = current.strip()
            idx = document.text.find(chunk_text, search_from)
            if idx == -1:
                idx = document.text.find(chunk_text)
            if idx == -1:
                idx = search_from
            start = idx
            end = start + len(chunk_text)
            search_from = start + max(1, len(chunk_text) // 2)
            chunk_hash = sha256(chunk_text.encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=f"{document.name}#{ordinal:04d}",
                    source=document.name,
                    heading=heading,
                    page=_page_for_offset(page_ranges, start, end),
                    start_char=start,
                    end_char=end,
                    text=chunk_text,
                    content_hash=chunk_hash,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    return chunks
