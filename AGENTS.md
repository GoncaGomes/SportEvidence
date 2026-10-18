# Repository Guidelines

## Project Structure & Module Organization

SportEvidence is an early-stage Python research assistant for sports-science evidence. Retrieval-augmented generation and PostgreSQL/pgvector storage are planned; current code focuses on corpus acquisition and parsing.

- `src/sportevidence/`: application package, including `cli.py`, the configuration placeholder `config.py`, and document conversion in `ingestion/parsing.py`.
- `scripts/download_corpus.py`: corpus acquisition, document identity checks, checksums, and download reports.
- `tests/test_download_corpus.py`: offline regression tests using generated payloads.
- `pyproject.toml` and `uv.lock`: package metadata, dependencies, development tools, and locked versions.

## Build, Test, and Development Commands

Run commands from the repository root with Python 3.12 or newer:

```powershell
uv sync --group dev       # Install application and development dependencies
uv run sportevidence     # Run the current placeholder CLI
uv run pytest            # Run the test suite
uv run ruff check .      # Check Python lint rules
uv run ruff format .     # Format Python files
uv build                 # Build source and wheel distributions
```

NEVER read the cwd .env file.
For corpus work, use `uv run python scripts/download_corpus.py`; add `--verify-only` for offline verification. These commands require the locally supplied `tests/corpus_desporto_50.json`; verification also requires existing downloads and their report under `tests/fixtures/`. Both locations are excluded from Git.

## Coding Style & Naming Conventions

Use four-space indentation, double-quoted strings, and Ruff formatting. Follow existing Python naming: `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Add type hints to new public interfaces. Keep reusable application logic under `src/sportevidence/` and operational scripts under `scripts/`.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions. Keep regression tests offline; use `tmp_path`, `monkeypatch`, and generated payloads rather than live downloads. Cover document identity, malformed payloads, and checksum failures when changing acquisition logic. Run focused checks with `uv run pytest tests/test_download_corpus.py`. No coverage threshold is currently configured.

## Commit & Pull Request Guidelines

History uses short, descriptive subjects, including `add corpus download and report` and `chore: add entry point, config file and updated README`; no uniform prefix convention is established. Keep commits focused. Pull requests should describe the resulting behavior, link relevant issues, and list validation commands and results. For corpus changes, explain provenance and verification effects; preserve catalogue IDs such as `SPORT-001`, source metadata, and checksum records.
