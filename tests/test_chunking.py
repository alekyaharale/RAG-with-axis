from rag_qa.chunking import chunk_document
from rag_qa.documents import SourceDocument, Page


def make_doc(name: str, text: str, pages=None) -> SourceDocument:
    if pages is None:
        pages = (Page(number=None, text=text),)
    return SourceDocument(
        name=name,
        path=None,  # type: ignore[arg-type]
        pages=pages,
        text=text,
        content_hash="hash",
    )


def test_chunking_heading_paths_and_fences():
    text = """# Title

Intro paragraph.

## Section A

Paragraph A1.

```python
# not a heading
print('hi')
```

### Subsection

Paragraph A2.

## Section B

Paragraph B.
"""
    doc = make_doc("doc.md", text)
    chunks = chunk_document(doc, chunk_chars=200, chunk_overlap=0)
    assert chunks
    headings = [c.heading for c in chunks]
    assert any("Title > Section A" in h for h in headings)
    assert any("Subsection" in h for h in headings)
    assert any("Section B" in h for h in headings)
    # code fence content present and not treated as heading
    all_text = "\n".join(c.text for c in chunks)
    assert "# not a heading" in all_text


def test_chunking_overlap_and_offsets():
    para = "Sentence one. Sentence two. Sentence three. Sentence four."
    text = f"# H\n\n{para}\n\n{para}\n\n{para}"
    doc = make_doc("doc.md", text)
    chunks = chunk_document(doc, chunk_chars=60, chunk_overlap=20)
    assert len(chunks) >= 2
    # offsets reconstruct text
    for c in chunks:
        assert doc.text[c.start_char:c.end_char] == c.text
    # stable ids
    assert chunks[0].chunk_id == "doc.md#0000"


def test_chunking_short_and_unicode():
    text = "# Ünïcødé\n\nHello 世界. This is a short document."
    doc = make_doc("u.md", text)
    chunks = chunk_document(doc, chunk_chars=1000, chunk_overlap=100)
    assert len(chunks) == 1
    assert "世界" in chunks[0].text
    assert chunks[0].heading.startswith("Ünïcødé")
