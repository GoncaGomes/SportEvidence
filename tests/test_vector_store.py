from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from sportevidence.storage import vector_store


def test_query_collection_preserves_all_queries_and_chunk_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    embed = Mock(return_value=[[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(vector_store, "embed_texts", embed)
    collection = Mock()
    collection.query.return_value = {
        "ids": [["a#0001", "b#0001"], ["c#0001"]],
        "documents": [["First", "Second"], ["Third"]],
        "embeddings": [[[1.0, 0.0], [0.8, 0.2]], [[0.0, 1.0]]],
        "metadatas": [
            [
                {"document_id": "a", "source_path": "a.pdf"},
                {"document_id": "b", "source_path": "b.pdf"},
            ],
            [{"document_id": "c", "source_path": "c.pdf"}],
        ],
        "distances": [[0.0, 0.2], [0.1]],
    }
    # `with chromadb.PersistentClient(...)` binds to `__enter__`; point it at
    # the configured client mock.
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_collection.return_value = collection
    monkeypatch.setattr(
        vector_store.chromadb, "PersistentClient", Mock(return_value=client)
    )

    result = vector_store.query_collection(["One", "Two"], tmp_path, "evidence")

    assert result.queries == ["One", "Two"]
    assert [
        [chunk.document_id for chunk in group] for group in result.retrieved_chunks
    ] == [["a", "b"], ["c"]]
    second = result.retrieved_chunks[0][1]
    assert (second.text, second.source_path, second.embedding, second.distance) == (
        "Second",
        "b.pdf",
        [0.8, 0.2],
        0.2,
    )
    assert result.retrieved_chunks[1][0].text == "Third"


def test_query_collection_keeps_empty_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vector_store, "embed_texts", Mock(return_value=[[1.0]]))
    client = MagicMock()
    client.__enter__.return_value = client
    client.get_collection.return_value.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "embeddings": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    monkeypatch.setattr(
        vector_store.chromadb, "PersistentClient", Mock(return_value=client)
    )

    result = vector_store.query_collection(["No match"], tmp_path, "evidence")

    assert result.queries == ["No match"]
    assert result.retrieved_chunks == [[]]
