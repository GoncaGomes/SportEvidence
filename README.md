# SportEvidence

SportEvidence indexes scientific PDFs and retrieves relevant passages with source references. It uses Docling to extract text and tables, an OpenAI-compatible endpoint to generate embeddings, and ChromaDB to store and search them locally. Sports science is the initial use case, and the same workflow can be used with other document collections.

The current workflow supports single-PDF ingestion through the CLI and semantic search through Python. The legacy chatbot code is retained for reference and awaits integration with the current retrieval pipeline.

## Quick start

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/getting-started/installation/). Run the commands below from the repository root.

```sh
uv sync --locked
uv run sportevidence --help
```

Set your embedding credentials in the environment or in a local `.env` file:

```dotenv
SKYNET_API_KEY=your-api-key
EMBEDDING_MODEL=your-embedding-model
```

Set `SKYNET_BASE_URL` if you use a custom OpenAI-compatible endpoint. `EMBEDDING_TIMEOUT_S` defaults to 30 seconds.

### Index a PDF

Replace `paper.pdf` with the path to a local PDF. `sports-science` is the destination collection.

```sh
uv run sportevidence ingest-to-db "paper.pdf" sports-science --db-path tests/fixtures/data --artifacts-dir tests/fixtures/data/ingestion
```

The command prepares the PDF, converts it to Docling JSON, associates table captions, splits the document into passages, embeds those passages, and writes them to ChromaDB. On success, it prints the document ID and the number of indexed chunks.

New collections created by the CLI use cosine distance. If `--db-path` is omitted, the command uses `SPORTEVIDENCE_DB_PATH` or falls back to `tests/fixtures/data`. The default artefact directory is `tests/fixtures/data/ingestion`.

## Document identity and source references

- The **document ID** is the SHA-256 of the original PDF. Moving or renaming a byte-identical PDF preserves its ID. A different PDF with the same filename receives a different ID.
- Each **chunk ID** combines the document ID with its zero-based position, in the format `<pdf_sha256>:<chunk_index>`.
- The **source path and source hash** refer to the Docling JSON used to generate the chunks. Page numbers refer to the prepared PDF, which is saved alongside that JSON.

Use a fresh collection when changing embedding models or rebuilding an index created with an earlier ID scheme.

## Optional corpus acquisition

The sports-science download utility uses a local catalogue at `tests/corpus_desporto_50.json`. With the catalogue in place, run:

```sh
uv run python scripts/download_corpus.py
```

To check existing downloads against the catalogue and recorded SHA-256 checksums offline, run:

```sh
uv run python scripts/download_corpus.py --verify-only
```

## Development

Run the same formatting, linting, and test checks as CI:

```sh
uv sync --locked --group dev
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest -q
```

The test suite covers PDF preparation, chunking, embedding request handling, CLI orchestration, and storage and retrieval with in-memory Chroma collections. It runs with test fixtures and mocked embedding requests.
