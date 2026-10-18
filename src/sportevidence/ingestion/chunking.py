import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sportevidence.ingestion.parsing import parse_markdown
from sportevidence.models import Chunk, Document

INPUT_FILE_TEST = Path("tests/fixtures/SPORT-001.pdf")


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 2500,
    overlap: int = 250,
) -> list[Chunk]:

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
    temp = []
    for i in range(len(chunks)):
        chunk = Chunk(
            chunk_id=document_id[0:5] + f"#{i + 1:04d}",
            text=chunks[i],
        )
        temp.append(chunk)
    return temp


def chunk_document(pdf_path: Path) -> Document:
    document_id = calculate_sha256(pdf_path)
    chunk_list = chunk_text(
        text=parse_markdown(pdf_path),
        document_id=document_id,
    )
    doc = Document(
        document_id=document_id, source_path=pdf_path, chunks=tuple(chunk_list)
    )
    return doc


def calculate_sha256(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


if __name__ == "__main__":
    document = chunk_document(INPUT_FILE_TEST)

    print(f"Document ID: {document.document_id}")
    print(f"Source: {document.source_path}")
    print(f"Number of chunks: {len(document.chunks)}")

    first_chunk = document.chunks[0]

    print(f"First chunk ID: {first_chunk.chunk_id}")
    print(f"Text sample: {first_chunk.text[:100]}")
    print(f"Embedding: {first_chunk.embedding}")
