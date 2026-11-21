from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest
from chromadb import Collection

from sportevidence.models import Chunk, Document
from sportevidence.storage.vector_store import (
    get_document,
    search_chunks,
    upsert_document,
)


@pytest.fixture
def collection() -> Iterator[Collection]:
    with chromadb.Client() as client:
        name = f"sports-{uuid4().hex}"
        collection = client.create_collection(
            name=name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        try:
            yield collection
        finally:
            client.delete_collection(name)


@pytest.fixture
def document() -> Document:
    return Document(
        document_id="training",
        title="Strength training",
        source_path=Path("papers/training.pdf"),
        source_hash="training-hash",
        chunks=[
            Chunk(
                chunk_id="training#0000",
                chunk_index=0,
                text="Strength improves with progressive loading.",
                page_numbers=(2, 3),
                embedding=[1.0, 0.0],
            ),
        ],
    )


def test_upsert_document_preserves_data_and_makes_it_searchable(
    collection: Collection, document: Document
) -> None:
    assert upsert_document(collection, document) == 1
    assert get_document(collection, document.document_id) == document

    (hit,) = search_chunks(collection, [1.0, 0.0])
    chunk = document.chunks[0]
    assert hit.chunk_id == chunk.chunk_id
    assert hit.document_id == document.document_id
    assert hit.text == chunk.text
    assert hit.source_path == document.source_path
    assert hit.page_numbers == chunk.page_numbers
    assert hit.distance == pytest.approx(0.0)


def test_upsert_document_updates_without_duplicating(
    collection: Collection, document: Document
) -> None:
    upsert_document(collection, document)
    updated = replace(
        document,
        title="Updated strength training",
        source_hash="updated-hash",
        chunks=[
            replace(
                document.chunks[0],
                text="Updated training guidance.",
                page_numbers=(4,),
                embedding=[0.0, 1.0],
            ),
        ],
    )

    assert upsert_document(collection, updated) == 1
    assert collection.count() == 1
    assert get_document(collection, document.document_id) == updated


def test_upsert_document_rejects_missing_embedding_without_saving_any_chunks(
    collection: Collection, document: Document
) -> None:
    incomplete = replace(
        document,
        chunks=[
            *document.chunks,
            Chunk(chunk_id="training#0001", chunk_index=1, text="Missing embedding."),
        ],
    )

    with pytest.raises(ValueError, match=r"Chunk 'training#0001' has no embedding"):
        upsert_document(collection, incomplete)

    assert collection.count() == 0


def test_search_chunks_orders_by_distance_and_respects_limit(
    collection: Collection, document: Document
) -> None:
    ranked = replace(
        document,
        chunks=[
            Chunk("training#0000", 0, "Farthest", embedding=[-1.0, 0.0]),
            Chunk("training#0001", 1, "Middle", embedding=[0.0, 1.0]),
            Chunk("training#0002", 2, "Closest", embedding=[1.0, 0.0]),
        ],
    )
    upsert_document(collection, ranked)

    hits = search_chunks(collection, [1.0, 0.0], limit=2)

    assert [hit.chunk_id for hit in hits] == ["training#0002", "training#0001"]
    assert [hit.distance for hit in hits] == pytest.approx([0.0, 1.0])


def test_search_chunks_filters_by_document_id(
    collection: Collection, document: Document
) -> None:
    other = Document(
        document_id="recovery",
        source_path=Path("papers/recovery.pdf"),
        source_hash="recovery-hash",
        chunks=[
            Chunk("recovery#0000", 0, "Recovery after exercise.", embedding=[0.0, 1.0]),
        ],
    )
    upsert_document(collection, document)
    upsert_document(collection, other)

    hits = search_chunks(collection, [1.0, 0.0], document_id=other.document_id)

    assert [(hit.document_id, hit.chunk_id) for hit in hits] == [
        (other.document_id, other.chunks[0].chunk_id)
    ]


def test_missing_document_and_unmatched_search_filter(
    collection: Collection, document: Document
) -> None:
    upsert_document(collection, document)

    assert get_document(collection, "missing") is None
    assert search_chunks(collection, [1.0, 0.0], document_id="missing") == []
