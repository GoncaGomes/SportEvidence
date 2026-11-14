import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from sportevidence.config import EmbeddingSettings
from sportevidence.ingestion import embeddings
from sportevidence.models import Chunk, Document


@pytest.fixture
def client() -> Mock:
    return Mock()


@pytest.fixture
def settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        model="test-embedding-model",
        api_key="test-key",
    )


def set_response(client: Mock, items: list[tuple[int, list[float]]]) -> None:
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in items]
    )


def test_empty_entry_needs_no_settings_or_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_settings = Mock(side_effect=AssertionError("Settings must not be loaded"))
    create_client = Mock(side_effect=AssertionError("Client must not be created"))
    monkeypatch.setattr(EmbeddingSettings, "from_env", load_settings)
    monkeypatch.setattr(embeddings, "OpenAI", create_client)

    assert embeddings.embed_texts([]) == []
    load_settings.assert_not_called()
    create_client.assert_not_called()


@pytest.mark.parametrize("blank", ["", " \n\t "])
def test_blank_text_is_rejected(
    client: Mock, settings: EmbeddingSettings, blank: str
) -> None:
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        embeddings.embed_texts(["Valid text", blank], client=client, settings=settings)

    client.embeddings.create.assert_not_called()


def test_embed_texts_orders_results_and_preserves_supplied_client(
    client: Mock, settings: EmbeddingSettings
) -> None:
    set_response(client, [(1, [-0.3, 0.4]), (0, [0.1, 0.2])])

    result = embeddings.embed_texts(
        ["First", "Second"], client=client, settings=settings
    )

    assert result == [[0.1, 0.2], [-0.3, 0.4]]
    client.embeddings.create.assert_called_once_with(
        model=settings.model, input=["First", "Second"], encoding_format="float"
    )
    client.close.assert_not_called()


def test_embed_texts_loads_settings_and_closes_owned_client(
    monkeypatch: pytest.MonkeyPatch, client: Mock, settings: EmbeddingSettings
) -> None:
    load_settings = Mock(return_value=settings)
    create_client = MagicMock()
    create_client.return_value.__enter__.return_value = client
    set_response(client, [(0, [0.1, 0.2])])
    monkeypatch.setattr(EmbeddingSettings, "from_env", load_settings)
    monkeypatch.setattr(embeddings, "OpenAI", create_client)

    assert embeddings.embed_texts(["First"]) == [[0.1, 0.2]]

    load_settings.assert_called_once_with()
    create_client.assert_called_once_with(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_s,
        max_retries=0,
    )
    create_client.return_value.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([], "indices"),
        ([(0, [1.0]), (0, [2.0])], "indices"),
        ([(0, [1.0]), (2, [2.0])], "indices"),
        ([(0, [1.0]), (1, [1.0, 2.0])], "dimensions"),
        ([(0, [float("nan")]), (1, [1.0])], "non-finite"),
        ([(0, [float("inf")]), (1, [1.0])], "non-finite"),
        ([(0, [float("-inf")]), (1, [1.0])], "non-finite"),
        ([(0, [0.0, 0.0]), (1, [1.0, 2.0])], "zero vector"),
        ([(0, []), (1, [])], "zero vector"),
    ],
    ids=[
        "missing",
        "duplicate",
        "unexpected",
        "dimensions",
        "nan",
        "infinity",
        "negative-infinity",
        "zero",
        "empty-vector",
    ],
)
def test_embed_texts_rejects_invalid_response(
    client: Mock,
    settings: EmbeddingSettings,
    items: list[tuple[int, list[float]]],
    message: str,
) -> None:
    set_response(client, items)

    with pytest.raises(RuntimeError, match=message):
        embeddings.embed_texts(["First", "Second"], client=client, settings=settings)


def test_embed_document_builds_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "paper.json"
    content = json.dumps({"name": "paper_prepared"}).encode("utf-8")
    json_path.write_bytes(content)
    chunk_document = Mock(return_value=(["First", "Second"], "Paper title"))
    embed_texts = Mock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr(embeddings, "chunk_document", chunk_document)
    monkeypatch.setattr(embeddings, "embed_texts", embed_texts)

    result = embeddings.embed_document(json_path)

    assert result == Document(
        title="Paper title",
        document_id="paper",
        source_path=json_path,
        source_hash=hashlib.sha256(content).hexdigest(),
        chunks=[
            Chunk(text="First", embedding=[0.1, 0.2]),
            Chunk(text="Second", embedding=[0.3, 0.4]),
        ],
    )
    chunk_document.assert_called_once_with(json_path)
    embed_texts.assert_called_once_with(["First", "Second"])


def test_save_embeded_document_json_creates_directory_and_serializes(
    tmp_path: Path,
) -> None:
    doc = Document(
        title=None,
        document_id="paper",
        source_path=tmp_path / "source.json",
        source_hash="source-hash",
        chunks=[Chunk(text="Evidence", embedding=[0.1, 0.2])],
    )
    output_directory = tmp_path / "nested" / "embedded"

    embeddings.save_embeded_document_json(doc, output_directory)

    assert json.loads(
        (output_directory / "paper.json").read_text(encoding="utf-8")
    ) == {
        "title": None,
        "document_id": "paper",
        "source_path": str(doc.source_path),
        "source_hash": "source-hash",
        "chunks": [{"text": "Evidence", "embedding": [0.1, 0.2]}],
    }


@pytest.mark.parametrize(
    "content", [b"", b"x" * (1024 * 1024 + 1)], ids=["empty", "multi-block"]
)
def test_calculate_sha256(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(content)

    assert embeddings.calculate_sha256(path) == hashlib.sha256(content).hexdigest()
