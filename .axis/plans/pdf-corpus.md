# Plan: Ship the sample corpus as PDF documents

**Slug:** `pdf-corpus` • **Created:** 2026-11-18 • **Base:** `3314f2d`

## Objective

`docs/corpus/` holds exactly three `.pdf` files, and ingestion extracts the same facts and heading citations from them as from the current Markdown samples. The loader keeps accepting `.md`, `.txt`, and `.pdf`.

## Context

- **Files involved:**
  - Corpus sources: `docs/corpus/01-product-overview.md`, `02-support-policy.md`, `03-operations-guide.md`.
  - Answer-key names: `eval/questions.json:6-7,17-18,29-30,40-41,51-53,63-64,75-76,86-87`.
  - Evaluation test names: `tests/test_evaluate.py:38-40`.
  - Corpus rules text: `README.md:8,88`.
  - Extraction and heading parsing: `rag_qa/documents.py:41-56`, `rag_qa/chunking.py:25,89-124`.
  - Corpus tests: `tests/test_documents.py:1-53`.
- **Constraints:**
  - The chunker recognizes headings only on `#`-prefixed lines (`chunking.py:25`). The generated PDFs must therefore carry the Markdown heading markers in their extracted text, so citations keep the `document > heading` shape. The markers will also be visible on the rendered page; the samples are ingestion fixtures, not presentation documents.
  - Exactly three files may sit in `docs/corpus/` (`documents.py:98-103`). The Markdown originals are deleted, not kept beside the PDFs.
  - A PDF must have an extractable text layer; OCR is unsupported (`documents.py:51-55`).
  - The loader's accepted types stay unchanged (`config.py:12`). Existing tests build Markdown corpora, and dropping that support is a separate decision.
- **Dependencies:** `pypdf>=7,<8` is already a runtime dependency (`pyproject.toml:15`). PDF generation needs a writer; `reportlab` is not a project dependency and is used only at generation time.

## Approach

**Chosen:** Render the three existing Markdown samples to PDF with ReportLab, keeping every line verbatim so the heading markers survive, verify extraction with `pypdf` before deleting the Markdown, then update the evaluation source names, the evaluation test names, and the README. The corpus changes format; ingestion code does not change.

**Alternative considered:** Set `SUPPORTED_EXTENSIONS` to `{".pdf"}` so only PDFs are accepted. That removes a capability the tests exercise and would require a PDF-writing helper inside the test suite. The request is about what the corpus contains, not what the loader may accept.

**Reasoning:** The corpus is data. Swapping its format exercises the existing PDF path (`documents.py:41-56`) without touching retrieval, chunking, or the agent contract. Keeping the Markdown markers in the text layer means `chunk_document` produces the same heading paths, so answers still cite a document and a heading.

**Left unbuilt:** font-size heading detection for PDFs, a committed corpus generator, and PDF-only enforcement in the loader. Font-size detection becomes worth building when real PDFs arrive without Markdown-style markers. A committed generator becomes worth building when the corpus is edited repeatedly. PDF-only enforcement becomes worth building if stray Markdown in `docs/corpus/` must fail ingestion rather than be ignored by the file count rule.

## Steps

1. `[new file: /tmp/build_corpus_pdf.py]` (scratch, not committed) Write a generator that reads the three `docs/corpus/*.md` files and writes `<stem>.pdf` beside each. Render with ReportLab `SimpleDocTemplate`, A4, 2 cm margins, and `Paragraph` styles: headings at 12–16 pt bold, body at 10 pt, each source line rendered verbatim so `#`, `##`, and `-` prefixes remain in the text. Escape `&`, `<`, `>`, and `(`/`)` for ReportLab markup, and skip the "Sample content to replace…" first line from the rendered body while keeping the rest. Run `uv run --no-project --with "reportlab>=4,<5" python /tmp/build_corpus_pdf.py`. Test impact: none; it writes three new binary files.
2. `[new file: docs/corpus/01-product-overview.pdf]` `[new file: docs/corpus/02-support-policy.pdf]` `[new file: docs/corpus/03-operations-guide.pdf]` (produced by step 1). Verify extraction before deleting sources with `uv run --no-project --with "pypdf>=7,<8" python /tmp/verify_corpus_pdf.py`, which asserts per file: the top-level `# ...` line, at least three `## ...` lines, and one known fact (`Nimbus Desk`, `15 minutes`, `first Saturday`). Print the extracted text so a human can inspect line breaks. Recovery: rerun step 1 while the `.md` files still exist.
3. `[docs/corpus/01-product-overview.md]` `[docs/corpus/02-support-policy.md]` `[docs/corpus/03-operations-guide.md]` Delete the three Markdown originals with `rm -f`. Recovery: `git restore docs/corpus/` restores them from `HEAD`.
4. `[eval/questions.json:6-7,17-18,29-30,40-41,51-53,63-64,75-76,86-87]` Change every `expected_sources` entry from `.md` to `.pdf`. Leave the unanswerable entries empty. Test impact: `test_load_questions_schema` still sees ten entries; `test_evaluate_questions_with_fake` needs step 5 to report nonzero recall.
5. `[tests/test_evaluate.py:38-40]` Rename the three `make_doc` sources to `01-product-overview.pdf`, `02-support-policy.pdf`, and `03-operations-guide.pdf` so the fake index matches the question file.
6. `[tests/test_documents.py:53]` Add `test_repo_corpus_is_three_pdf_documents`. It loads settings at the repo root, calls `load_corpus`, and asserts: exactly three names, every name ends in `.pdf`, every document text is non-empty, and every document contains at least one line matching `^#+ `. This is the regression test for the shipped corpus; unit tests that build temporary Markdown corpora stay unchanged.
7. `[README.md:8]` Replace "three sample Markdown documents" with "three sample PDF documents" and keep the instruction to replace them with the real references.
8. `[README.md:88]` Reword the corpus rule: the repository corpus holds three PDF documents; ingestion still accepts `.md`, `.txt`, and `.pdf`; PDFs need a text layer. Add one sentence that the sample PDFs carry Markdown heading markers so heading-aware chunking and citations work.

## Acceptance criteria

- [ ] `docs/corpus/` contains exactly three files, all ending in `.pdf`; no `.md` file remains there.
- [ ] `load_corpus` returns three documents, and each document's extracted text contains its `#` and `##` heading lines, so chunk headings read like `Product Overview > Purpose`.
- [ ] `eval/questions.json` names the `.pdf` files, and `tests/test_evaluate.py` uses the same names.
- [ ] `uv run pytest` passes without network access, including the new corpus regression test.
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [ ] The loader still accepts `.md` and `.txt`; existing document tests pass unchanged.
- [ ] README describes the PDF corpus and the retained supported types without claiming OCR support.

## Risks & rollback

- **Risk:** ReportLab output has no blank line between paragraphs, so chunk boundaries inside a section differ from the Markdown run. → **Mitigation:** Chunking still splits on headings and sentence boundaries, and retrieval evaluation scores source files, not chunk offsets. Verify every heading and known fact before deleting the Markdown.
- **Risk:** A heading wraps across lines and leaves `##` on a line without its title, so the heading is misread. → **Mitigation:** Headings are short and render at 12–16 pt on A4. The verification script asserts each heading line before sources are removed.
- **Risk:** A future `pypdf` change alters extracted spacing or drops the markers. → **Mitigation:** `pypdf` is pinned to `<8`, and the new corpus test fails when markers disappear.
- **Risk:** Binary PDFs are opaque in review. → **Mitigation:** The verification script prints extracted text, and the Markdown originals remain recoverable from `3314f2d`.
- **Rollback:** `git restore docs/corpus/` restores the Markdown files, then delete the three PDFs; revert the edits to `eval/questions.json`, `tests/test_evaluate.py`, `tests/test_documents.py`, and `README.md`.

## Out of scope

- Restricting `SUPPORTED_EXTENSIONS` to PDF only, or rejecting non-PDF files in `docs/corpus/`.
- OCR, scanned PDFs, or font-based heading detection.
- Committing a PDF generator or keeping the Markdown sources beside the PDFs.
- `tests/fixtures/corpus/*.md`; those fixture files are unreferenced and stay Markdown.
- Rebuilding the gitignored `data/index.json` or running the live evaluation, which need `GOOGLE_API_KEY`.
