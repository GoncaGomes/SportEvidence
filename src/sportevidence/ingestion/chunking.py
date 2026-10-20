import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sportevidence.ingestion.parsing import parse_markdown
from sportevidence.models import Chunk, Document


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 2500,
    overlap: int = 250,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap > chunk_size or overlap < 0:
        raise ValueError("overlap must satisfy 0 <= overlap <= chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        is_separator_regex=False,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [
        Chunk(
            chunk_id=f"{document_id}#{index:04d}",
            text=chunk_text,
        )
        for index, chunk_text in enumerate(chunks, start=1)
    ]


def chunk_document(pdf_path: Path) -> Document:
    document_id = calculate_sha256(pdf_path)
    chunk_list = chunk_text(
        text=parse_markdown(pdf_path),
        document_id=document_id,
    )
    return Document(
        document_id=document_id, source_path=pdf_path, chunks=tuple(chunk_list)
    )


def calculate_sha256(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
