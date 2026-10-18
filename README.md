# SportEvidence

A research assistant for exploring sports science through evidence-based answers.

SportEvidence is being developed to retrieve, compare, and synthesise findings across scientific documents using retrieval-augmented generation (RAG). The goal is to make research easier to explore while keeping answers traceable to their sources.

## Scope

The project covers running, strength training, athletics, swimming, football, tennis, and other sports, alongside cross-cutting topics such as training, recovery, nutrition, and supplementation.

## Current Capabilities

The project is in early-stage development, with corpus acquisition and document ingestion implemented:

- Download corpus documents, check their identity, and record source details and SHA-256 checksums in download reports.
- Verify existing downloads offline against the catalogue and recorded checksums.
- Convert PDFs to Markdown using PyMuPDF4LLM, validating file paths and rejecting empty conversion results.
- Split text into chunks with configurable character limits and overlap using LangChain's recursive text splitter.
- Represent results as immutable `Document` and `Chunk` dataclasses, retaining document IDs, chunk IDs, source paths, and text.

Retrieval, embeddings, persistent storage, and answer generation are planned work.

## Getting Started

Install Python 3.12 or newer and `uv`. Run these commands from the repository root:

```powershell
uv sync --group dev
uv run sportevidence
```

The CLI currently prints a placeholder greeting. Use the Python ingestion functions to parse and chunk documents.

## Document Ingestion

Run the following example in the project environment, such as a Python session started with `uv run python`. Replace `document.pdf` with an existing PDF path:

```python
from pathlib import Path

from sportevidence.ingestion.chunking import chunk_document

document = chunk_document(Path("document.pdf"))
print(f"Created {len(document.chunks)} chunks")
print(document.chunks[0].text)
```

`chunk_document` converts the PDF to Markdown, hashes the original file bytes with SHA-256 for the document ID, and returns a `Document` containing a tuple of chunks. Each chunk retains that document ID and the source path. Chunk IDs combine the first five characters of the document ID with a sequential suffix such as `#0001`.

The default chunk size is 2,500 characters with a target overlap of 250 characters. To configure these values for text directly, use `chunk_text(text, source_path, document_id, chunk_size=2500, overlap=250)` from the same module. To convert a PDF without chunking it, use `parse_markdown(path)` from `sportevidence.ingestion.parsing`.

## Corpus Acquisition

The initial corpus targets 50 credible documents published within the last five years, prioritising systematic reviews, meta-analyses, consensus statements, and relevant primary research. Source selection records publication details, links, and an inclusion rationale.

Supply the local catalogue at `tests/corpus_desporto_50.json` before running:

```powershell
uv run python scripts/download_corpus.py
```

The script downloads documents into `tests/fixtures/` and writes reports, including `tests/fixtures/relatorio_downloads.json`. It rechecks and reuses successful downloads, retries missing or failed entries, and preserves the bibliographic catalogue.

To verify downloaded files offline:

```powershell
uv run python scripts/download_corpus.py --verify-only
```

Verification requires the catalogue, existing downloads, and their JSON report. The catalogue and `tests/fixtures/` are excluded from Git and must be supplied locally.

## Project Structure

```text
src/sportevidence/
    cli.py                Placeholder command-line entry point
    config.py             Configuration placeholder
    models.py             Immutable Document and Chunk models
    ingestion/
        parsing.py        PDF-to-Markdown conversion
        chunking.py       Text splitting and document hashing
scripts/
    download_corpus.py    Corpus acquisition and verification
tests/
    test_download_corpus.py
    test_parsing.py
    test_chunking.py
    test_models.py
```

## Development and Testing

Run the full test suite or focus on chunking and models:

```powershell
uv run pytest
uv run pytest tests/test_chunking.py tests/test_models.py
```

Tests run offline using temporary files, generated payloads, and mocked dependencies. They cover corpus acquisition and verification, PDF parsing, chunk boundaries and overlap, hashing, document assembly, and model behaviour. The test suite does not require the local corpus catalogue or downloaded fixtures.

Check lint and formatting, apply formatting, or build distributions:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run ruff format .
uv build
```

## Planned Capabilities

- Enrich ingested documents with publication metadata and source references for retrieval.
- Search across multiple documents using semantic and keyword retrieval.
- Rerank retrieved passages to improve relevance.
- Generate answers with citations and identify conflicting findings or insufficient evidence.
- Evaluate retrieval quality, citation accuracy, and answer grounding.

## Technical Approach

The planned foundation is a modular Python pipeline with PostgreSQL and pgvector for persistent document, passage, and embedding storage. Retrieval will supply relevant evidence to a language model for synthesis.

Development will start with a measurable RAG baseline. Hierarchical retrieval, graph-based retrieval, and agent workflows will be explored once the core pipeline is validated.
