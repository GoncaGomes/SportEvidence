import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import openai
import pytest

from sportevidence.ingestion import chunking
from sportevidence.models import Chunk, Document


@pytest.fixture
def embeddings(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    client = MagicMock()
    client.__enter__.return_value = client
    factory = Mock(return_value=client)
    monkeypatch.setattr(openai, "OpenAI", factory)
    monkeypatch.setenv("SKYNET_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("SKYNET_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_S", "30")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding-model")
    spec = importlib.util.spec_from_file_location(
        "sportevidence.ingestion.embeddings",
        Path(chunking.__file__).with_name("embeddings.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory.assert_not_called()
    # Keep a reference to the fake client for response setup and assertions.
    module.CLIENT = client
    return module


def set_response(embeddings: ModuleType, items: list[tuple[int, list[float]]]) -> None:
    embeddings.CLIENT.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in items]
    )


def test_embed_texts_empty_input_skips_request(embeddings: ModuleType) -> None:
    assert embeddings.embed_texts([]) == []
    embeddings.CLIENT.embeddings.create.assert_not_called()


def test_embed_texts_accepts_a_caller_owned_client(embeddings: ModuleType) -> None:
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
    )

    assert embeddings.embed_texts(["First"], client=client) == [[0.1, 0.2]]
    client.close.assert_not_called()
    embeddings.OpenAI.assert_not_called()


def test_embed_texts_rejects_missing_model_before_creating_client(
    embeddings: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL")

    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        embeddings.embed_texts(["First"])

    embeddings.OpenAI.assert_not_called()


@pytest.mark.parametrize("texts", [[""], [" \n\t"], ["Valid text", " "]])
def test_embed_texts_rejects_blank_input(
    embeddings: ModuleType, texts: list[str]
) -> None:
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        embeddings.embed_texts(texts)

    embeddings.CLIENT.embeddings.create.assert_not_called()


def test_embed_texts_orders_response_and_passes_request_arguments(
    embeddings: ModuleType,
) -> None:
    set_response(embeddings, [(1, [0.3, 0.4]), (0, [0.1, -0.2])])

    assert embeddings.embed_texts(["First", "Second"]) == [[0.1, -0.2], [0.3, 0.4]]
    embeddings.CLIENT.embeddings.create.assert_called_once_with(
        model="test-embedding-model", input=["First", "Second"], encoding_format="float"
    )


@pytest.mark.parametrize("indices", [[], [0], [0, 0], [0, 2], [-1, 0], [0, 1, 2]])
def test_embed_texts_rejects_invalid_response_indices(
    embeddings: ModuleType, indices: list[int]
) -> None:
    set_response(embeddings, [(index, [0.1, 0.2]) for index in indices])

    with pytest.raises(RuntimeError, match="embedding indices"):
        embeddings.embed_texts(["First", "Second"])


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[0.1, 0.2], [0.3]], "dimensions do not match"),
        ([[float("nan")], [0.1]], "non-finite"),
        ([[0.1], [float("inf")]], "non-finite"),
        ([[float("-inf")], [0.1]], "non-finite"),
        ([[0.1, 0.2], [0.0, 0.0]], "zero vector"),
        ([[], []], "zero vector"),
    ],
    ids=["dimensions", "nan", "infinity", "negative-infinity", "zeros", "empty"],
)
def test_embed_texts_rejects_invalid_vectors(
    embeddings: ModuleType, vectors: list[list[float]], message: str
) -> None:
    set_response(embeddings, list(enumerate(vectors)))

    with pytest.raises(RuntimeError, match=message):
        embeddings.embed_texts(["First", "Second"])


def test_embed_texts_propagates_client_errors(embeddings: ModuleType) -> None:
    error = RuntimeError("Embedding service unavailable")
    embeddings.CLIENT.embeddings.create.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        embeddings.embed_texts(["First"])

    assert raised.value is error


@pytest.mark.parametrize("existing_embedding", [None, [0.9, 0.8]])
def test_embed_chunks_preserves_document_and_replaces_embeddings(
    embeddings: ModuleType, existing_embedding: list[float] | None
) -> None:
    chunks = (
        Chunk("abc12#0001", "First", existing_embedding),
        Chunk("abc12#0002", "Second"),
    )
    document = Document("abc123", Path("source.pdf"), chunks)
    set_response(embeddings, [(1, [0.3, 0.4]), (0, [0.1, -0.2])])

    result = embeddings.embed_chunks(document)

    assert result == Document(
        document.document_id,
        document.source_path,
        (
            Chunk("abc12#0001", "First", [0.1, -0.2]),
            Chunk("abc12#0002", "Second", [0.3, 0.4]),
        ),
    )
    assert result is not document
    assert all(new is not old for new, old in zip(result.chunks, chunks))
    assert document.chunks is chunks
    assert chunks[0].embedding == existing_embedding
    assert chunks[1].embedding is None
    embeddings.CLIENT.embeddings.create.assert_called_once_with(
        model="test-embedding-model", input=["First", "Second"], encoding_format="float"
    )


def test_embed_chunks_empty_document_skips_request(embeddings: ModuleType) -> None:
    document = Document("abc123", Path("source.pdf"), ())

    assert embeddings.embed_chunks(document) == document
    embeddings.CLIENT.embeddings.create.assert_not_called()


def test_embed_chunks_propagates_errors_without_changing_document(
    embeddings: ModuleType,
) -> None:
    document = Document("abc123", Path("source.pdf"), (Chunk("abc12#0001", "First"),))
    set_response(embeddings, [(0, [0.0])])

    with pytest.raises(RuntimeError, match="zero vector"):
        embeddings.embed_chunks(document)

    assert document.chunks[0].embedding is None
