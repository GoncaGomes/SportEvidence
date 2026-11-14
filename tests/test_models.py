from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sportevidence.models import Chunk, Document


@pytest.fixture
def chunk() -> Chunk:
    return Chunk("abc12#0001")


def test_chunk_preserves_fields(chunk: Chunk) -> None:
    assert chunk.text == "abc12#0001"
    assert chunk.embedding is None


def test_chunk_preserves_embedding() -> None:
    vector = [0.1, -0.2, 0.3]
    chunk = Chunk("abc12#0001", embedding=vector)

    assert chunk.embedding == vector


def test_document_preserves_chunk_order(chunk: Chunk) -> None:
    second = Chunk("Next text.")
    chunks = [chunk, second]

    document = Document(
        title="title",
        document_id="abc123",
        source_path=Path("source.pdf"),
        source_hash="hash",
        chunks=chunks,
    )

    assert document.chunks is chunks
    assert document.chunks == [chunk, second]
    assert document.document_id == "abc123"
    assert document.source_path == Path("source.pdf")


def test_document_accepts_empty_list() -> None:
    assert (
        Document(
            document_id="abc123",
            source_path=Path("source.pdf"),
            source_hash="hash",
            chunks=[],
        ).chunks
        == []
    )


def test_models_compare_by_value(chunk: Chunk) -> None:
    equal_chunk = Chunk("abc12#0001")
    different_chunk = Chunk("abc12#0002")

    assert chunk == equal_chunk
    assert chunk != different_chunk
    document = Document(
        document_id="abc123",
        source_path=Path("source.pdf"),
        source_hash="hash",
        chunks=[
            chunk,
        ],
    )
    assert document == Document(
        document_id="abc123",
        source_path=Path("source.pdf"),
        source_hash="hash",
        chunks=[
            equal_chunk,
        ],
    )
    assert document != Document(
        document_id="abc123",
        source_path=Path("source.pdf"),
        source_hash="hash",
        chunks=[
            different_chunk,
        ],
    )
    assert document != Document(
        document_id="abc123",
        source_path=Path("source.pdf"),
        source_hash="hash",
        chunks=[],
    )
    assert chunk != Chunk(chunk.text, [0.1])
    assert document != Document(
        document_id="other",
        source_path=document.source_path,
        source_hash="hash1",
        chunks=document.chunks,
    )
    assert document != Document(
        document_id=document.document_id,
        source_path=Path("other.pdf"),
        chunks=document.chunks,
        source_hash="hash3",
    )


@pytest.mark.parametrize("field", ["text", "embedding"])
def test_chunk_fields_are_frozen(chunk: Chunk, field: str) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(chunk, field, "replacement")


@pytest.mark.parametrize(
    "field", ["document_id", "source_path", "chunks", "source_hash", "title"]
)
def test_document_fields_are_frozen(chunk: Chunk, field: str) -> None:
    document = Document(
        title="Title1",
        document_id="abc123",
        source_path=Path("abc123"),
        source_hash="hash",
        chunks=[
            chunk,
        ],
    )

    with pytest.raises(FrozenInstanceError):
        setattr(document, field, "replacement")


def test_models_use_slots(chunk: Chunk) -> None:
    assert not hasattr(chunk, "__dict__")
    assert not hasattr(
        Document(
            title="Title1",
            document_id="abc123",
            source_path=Path("abc123"),
            source_hash="hash",
            chunks=[
                chunk,
            ],
        ),
        "__dict__",
    )
