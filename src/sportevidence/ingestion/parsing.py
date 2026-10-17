from pathlib import Path

import pymupdf4llm


def _validate_corpus_file(path: Path) -> Path:

    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path}")

    return path


def parse_markdown(path: Path) -> str:
    """Convert a PDF file into a Markdown string using pymupdf4llm.

    Args:
        path: The path to the PDF file.

    Returns:
        A Markdown string representation of the PDF file.
    """
    path = _validate_corpus_file(path)
    file_md = pymupdf4llm.to_markdown(path)

    if file_md.strip() == "":
        raise ValueError(f"Empty parse result from PDF file: {path}")

    return file_md
