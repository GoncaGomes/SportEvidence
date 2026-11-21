from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from sportevidence.models import Chunk, Document, SearchHit


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(chunk_id="abc123:0", chunk_index=0, text="First text.")


def test_chunk_preserves_fields(chunk: Chunk) -> None:
    assert chunk.chunk_id == "abc123:0"
    assert chunk.chunk_index == 0
    assert chunk.text == "First text."
    assert chunk.page_numbers == ()
    assert chunk.embedding is None


def test_chunk_preserves_embedding_and_pages() -> None:
    vector = [0.1, -0.2, 0.3]
    chunk = Chunk(
        chunk_id="abc123:1",
        chunk_index=1,
        text="Next text.",
        page_numbers=(2, 3),
        embedding=vector,
    )

    assert chunk.chunk_id == "abc123:1"
    assert chunk.chunk_index == 1
    assert chunk.text == "Next text."
    assert chunk.page_numbers == (2, 3)
    assert chunk.embedding == vector


def test_document_preserves_chunk_order(chunk: Chunk) -> None:
    second = Chunk(chunk_id="abc123:1", chunk_index=1, text="Next text.")
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
    assert document.source_hash == "hash"
    assert document.title == "title"


def test_document_accepts_empty_list() -> None:
    document = Document(
        document_id="abc123",
        source_path=Path("source.pdf"),
        source_hash="hash",
        chunks=[],
    )
    assert document.chunks == []
    assert document.title is None


@pytest.mark.parametrize(
    "title", ["Training outcomes", None], ids=["titled", "untitled"]
)
def test_search_hit_preserves_fields(title: str | None) -> None:
    source_path = Path("source.pdf")
    hit = SearchHit(
        chunk_id="abc123:0",
        document_id="abc123",
        title=title,
        text="First text.",
        source_path=source_path,
        page_numbers=(2, 3),
        distance=0.25,
    )

    assert hit.chunk_id == "abc123:0"
    assert hit.document_id == "abc123"
    assert hit.title == title
    assert hit.text == "First text."
    assert hit.source_path == source_path
    assert hit.page_numbers == (2, 3)
    assert hit.distance == 0.25


def test_models_compare_by_value(chunk: Chunk) -> None:
    equal_chunk = Chunk(chunk_id="abc123:0", chunk_index=0, text="First text.")
    different_chunk = replace(chunk, text="Different text.")

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
    assert document != replace(document, document_id="other")
    assert document != replace(document, source_path=Path("other.pdf"))
    assert document != replace(document, source_hash="other-hash")
    assert document != replace(document, title="Other title")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", "abc123:1"),
        ("chunk_index", 1),
        ("text", "Different text."),
        ("page_numbers", (2, 3)),
        ("embedding", [0.1]),
    ],
)
def test_chunk_equality_includes_every_field(
    chunk: Chunk, field: str, value: object
) -> None:
    assert chunk != replace(chunk, **{field: value})


@pytest.mark.parametrize(
    "field", ["chunk_id", "chunk_index", "text", "page_numbers", "embedding"]
)
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
