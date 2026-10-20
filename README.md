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
- Represent documents and retrieval results with dataclasses, retaining document IDs, chunk IDs, source paths, and text.
- Generate embeddings through an OpenAI-compatible endpoint and store documents in persistent Chroma collections using cosine distance.
- Retrieve passages for one or more queries through the CLI, or use a configured Python query tool that returns source references without embedding vectors.

Answer generation and agent orchestration are planned work.

## Getting Started

Install Python 3.12 or newer and `uv`. Run these commands from the repository root:

```powershell
uv sync --group dev
uv run sportevidence --help
```

The CLI provides `document-to-db` and `query-db` commands. Help does not require embedding credentials.

Set `SKYNET_API_KEY` and `EMBEDDING_MODEL` in the process environment before ingesting or querying. Set `SKYNET_BASE_URL` for a custom OpenAI-compatible endpoint; otherwise the SDK uses its default endpoint. `EMBEDDING_TIMEOUT_S` defaults to 30 seconds and must be finite and positive. Configuration does not automatically load `.env` files.

```powershell
uv run sportevidence document-to-db document.pdf --input-db-path data/chroma --collection-name evidence
uv run sportevidence query-db data/chroma evidence --query "What supports recovery?" --n-results 5
```

Repeat `--query` to search multiple queries in one call. Results include text, document and chunk IDs, source paths, and cosine distances (lower is closer).

Ingestion defaults to `data/chroma` and `default_collection`. Override these defaults with `SPORTEVIDENCE_DB_PATH` and `SPORTEVIDENCE_COLLECTION`, or the CLI options above. Use the same embedding model for ingestion and queries. Rebuild the collection when changing models, even when vector dimensions match.

## Document Ingestion

Run the following example in the project environment, such as a Python session started with `uv run python`. Replace `document.pdf` with an existing PDF path:

```python
from pathlib import Path

from sportevidence.ingestion.chunking import chunk_document

document = chunk_document(Path("document.pdf"))
print(f"Created {len(document.chunks)} chunks")
print(document.chunks[0].text)
```

`chunk_document` converts the PDF to Markdown, hashes the original file bytes with SHA-256 for the document ID, and returns a `Document` containing a tuple of chunks. The document retains the source path. Chunk IDs combine the full document hash with a sequential suffix such as `#0001`; storage records the document ID and source path alongside each chunk.

Collections created with the previous five-character hash IDs must be rebuilt into a fresh collection and ingested again. Upserting with the new IDs into an old collection leaves the old records in place.

The default chunk size is 2,500 characters with a target overlap of 250 characters. To configure these values for text directly, use `chunk_text(text, document_id, chunk_size=2500, overlap=250)` from the same module. To convert a PDF without chunking it, use `parse_markdown(path)` from `sportevidence.ingestion.parsing`.

## Query Tool

Bind database settings in the application before exposing `tool.query` to an agent. Its arguments are query text and result count; its result is JSON-serializable and includes passages, source references, and distance, without vectors.

```python
from pathlib import Path

from sportevidence.config import RetrievalSettings
from sportevidence.query import QueryTool

tool = QueryTool(RetrievalSettings(Path("data/chroma"), "evidence"))
result = tool.query("What supports recovery?", n_results=5)
print(result["passages"])
```

`QueryTool()` uses the retrieval environment settings described above. Blank queries and nonpositive result counts raise `ValueError`; storage and embedding failures propagate to the caller. `embed_texts` and `embed_chunks` accept optional `client` and `settings` keyword arguments for injecting an embedding client and `EmbeddingSettings`. Injected clients remain owned by the caller.

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
    cli.py                Document ingestion and query commands
    config.py             Explicit environment settings
    models.py             Document, chunk, and retrieval models
    query.py              Configured query tool returning source passages
    ingestion/
        parsing.py        PDF-to-Markdown conversion
        chunking.py       Text splitting and document hashing
        embeddings.py     Embedding requests and response validation
    storage/
        vector_store.py   Chroma document upserts and batched retrieval
scripts/
    download_corpus.py    Corpus acquisition and verification
tests/
    test_download_corpus.py
    test_parsing.py
    test_chunking.py
    test_models.py
    test_embeddings.py
    test_vector_store.py
    test_query.py
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
- Add keyword retrieval alongside semantic search.
- Rerank retrieved passages to improve relevance.
- Generate answers with citations and identify conflicting findings or insufficient evidence.
- Evaluate retrieval quality, citation accuracy, and answer grounding.

## Technical Approach

The current pipeline uses Chroma for persistent passage and embedding storage. PostgreSQL and pgvector remain a possible future storage direction. Retrieval will supply relevant evidence to a language model for synthesis.
