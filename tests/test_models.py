from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sportevidence.models import Chunk, Document


@pytest.fixture
def chunk() -> Chunk:
    return Chunk("abc123", "abc12#0001", Path("source.pdf"), "Example text.")


def test_chunk_preserves_fields(chunk: Chunk) -> None:
    assert chunk.document_id == "abc123"
    assert chunk.chunk_id == "abc12#0001"
    assert chunk.source_path == Path("source.pdf")
    assert chunk.text == "Example text."


def test_document_preserves_chunk_order(chunk: Chunk) -> None:
    second = Chunk("abc123", "abc12#0002", Path("source.pdf"), "Next text.")
    chunks = (chunk, second)

    document = Document(chunks)

    assert document.chunks is chunks
    assert document.chunks == (chunk, second)


def test_document_accepts_empty_tuple() -> None:
    assert Document(()).chunks == ()


def test_models_compare_by_value(chunk: Chunk) -> None:
    equal_chunk = Chunk("abc123", "abc12#0001", Path("source.pdf"), "Example text.")
    different_chunk = Chunk("abc123", "abc12#0002", Path("source.pdf"), "Next text.")

    assert chunk == equal_chunk
    assert chunk != different_chunk
    assert Document((chunk,)) == Document((equal_chunk,))
    assert Document((chunk,)) != Document((different_chunk,))
    assert Document((chunk,)) != Document(())


@pytest.mark.parametrize("field", ["document_id", "chunk_id", "source_path", "text"])
def test_chunk_fields_are_frozen(chunk: Chunk, field: str) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(chunk, field, "replacement")


def test_document_chunks_are_frozen(chunk: Chunk) -> None:
    document = Document((chunk,))

    with pytest.raises(FrozenInstanceError):
        document.chunks = ()


def test_models_use_slots(chunk: Chunk) -> None:
    assert not hasattr(chunk, "__dict__")
    assert not hasattr(Document((chunk,)), "__dict__")
