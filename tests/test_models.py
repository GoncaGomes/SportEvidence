from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sportevidence.models import Chunk, Document


@pytest.fixture
def chunk() -> Chunk:
    return Chunk("abc12#0001", "Example text.")


def test_chunk_preserves_fields(chunk: Chunk) -> None:
    assert chunk.chunk_id == "abc12#0001"
    assert chunk.text == "Example text."
    assert chunk.embedding is None


def test_chunk_preserves_embedding() -> None:
    vector = [0.1, -0.2, 0.3]
    chunk = Chunk("abc12#0001", "Example text.", embedding=vector)

    assert chunk.embedding == vector


def test_document_preserves_chunk_order(chunk: Chunk) -> None:
    second = Chunk("abc12#0002", "Next text.")
    chunks = (chunk, second)

    document = Document("abc123", Path("source.pdf"), chunks)

    assert document.chunks is chunks
    assert document.chunks == (chunk, second)
    assert document.document_id == "abc123"
    assert document.source_path == Path("source.pdf")


def test_document_accepts_empty_tuple() -> None:
    assert Document("abc123", Path("source.pdf"), ()).chunks == ()


def test_models_compare_by_value(chunk: Chunk) -> None:
    equal_chunk = Chunk("abc12#0001", "Example text.")
    different_chunk = Chunk("abc12#0002", "Next text.")

    assert chunk == equal_chunk
    assert chunk != different_chunk
    document = Document("abc123", Path("source.pdf"), (chunk,))
    assert document == Document("abc123", Path("source.pdf"), (equal_chunk,))
    assert document != Document("abc123", Path("source.pdf"), (different_chunk,))
    assert document != Document("abc123", Path("source.pdf"), ())
    assert chunk != Chunk(chunk.chunk_id, chunk.text, [0.1])
    assert document != Document("other", document.source_path, document.chunks)
    assert document != Document(
        document.document_id, Path("other.pdf"), document.chunks
    )


@pytest.mark.parametrize("field", ["chunk_id", "text", "embedding"])
def test_chunk_fields_are_frozen(chunk: Chunk, field: str) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(chunk, field, "replacement")


@pytest.mark.parametrize("field", ["document_id", "source_path", "chunks"])
def test_document_fields_are_frozen(chunk: Chunk, field: str) -> None:
    document = Document("abc123", Path("source.pdf"), (chunk,))

    with pytest.raises(FrozenInstanceError):
        setattr(document, field, "replacement")


def test_models_use_slots(chunk: Chunk) -> None:
    assert not hasattr(chunk, "__dict__")
    assert not hasattr(Document("abc123", Path("source.pdf"), (chunk,)), "__dict__")
