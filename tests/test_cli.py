from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from sportevidence import cli


@pytest.fixture(autouse=True)
def dotenv_loader(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Keep CLI configuration independent of the local environment."""
    loader = Mock(return_value=False)
    monkeypatch.setattr(cli, "load_dotenv", loader)
    monkeypatch.setattr(cli.RetrievalSettings, "from_env", Mock())
    return loader


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Avoid real PDF processing, network requests, and database or file writes."""
    mocks = SimpleNamespace(
        calculate_sha256=Mock(),
        fix_pdf_orientation=Mock(),
        convert_document=Mock(),
        associate_table_captions=Mock(),
        embed_document=Mock(),
        save_embeded_document_json=Mock(),
        upsert_document=Mock(),
    )
    for name, service in vars(mocks).items():
        monkeypatch.setattr(cli, name, service)
    mocks.persistent_client = Mock()
    monkeypatch.setattr(cli.chromadb, "PersistentClient", mocks.persistent_client)
    return mocks


def test_help_needs_no_embedding_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dotenv_loader: Mock,
) -> None:
    monkeypatch.delenv("SKYNET_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    assert raised.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "fix-pdf-orientation",
        "convert-pdf-docling",
        "associate-captions",
        "embed-and-save-document",
        "ingest-to-db",
    ):
        assert command in output
    dotenv_loader.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["unknown-command"],
        ["fix-pdf-orientation", "source.pdf"],
        ["convert-pdf-docling"],
        ["associate-captions"],
        ["embed-and-save-document"],
        ["convert-pdf-docling", "source.pdf", "--unknown-option"],
        ["ingest-to-db"],
        ["ingest-to-db", "source.pdf"],
    ],
    ids=[
        "no-command",
        "unknown-command",
        "missing-output-pdf",
        "missing-input-pdf",
        "missing-caption-input",
        "missing-embedding-input",
        "unknown-option",
        "missing-ingestion-input",
        "missing-collection-name",
    ],
)
def test_invalid_arguments_skip_services(
    arguments: list[str], services: SimpleNamespace, dotenv_loader: Mock
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)

    assert raised.value.code == 2
    dotenv_loader.assert_not_called()
    for service in vars(services).values():
        service.assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "service_name", "paths"),
    [
        (
            ["fix-pdf-orientation", "source.pdf", "fixed.pdf"],
            "fix_pdf_orientation",
            [Path("source.pdf"), Path("fixed.pdf")],
        ),
        (
            ["associate-captions", "source.json"],
            "associate_table_captions",
            [Path("source.json")],
        ),
    ],
)
def test_processing_commands_pass_paths_and_return_success(
    arguments: list[str],
    service_name: str,
    paths: list[Path],
    services: SimpleNamespace,
    dotenv_loader: Mock,
) -> None:
    calls = Mock()
    calls.attach_mock(dotenv_loader, "load_dotenv")
    for name, service in vars(services).items():
        calls.attach_mock(service, name)

    assert cli.main(arguments) == 0
    assert calls.mock_calls == [
        call.load_dotenv(),
        getattr(call, service_name)(*paths),
    ]


@pytest.mark.parametrize(
    ("options", "output_path"),
    [
        ([], Path("tests/fixtures/corpus_json")),
        (["--output-json-path", "converted"], Path("converted")),
    ],
    ids=["default-output", "custom-output"],
)
def test_conversion_passes_output_path(
    services: SimpleNamespace, options: list[str], output_path: Path
) -> None:
    assert cli.main(["convert-pdf-docling", "source.pdf", *options]) == 0

    services.convert_document.assert_called_once_with(Path("source.pdf"), output_path)


@pytest.mark.parametrize(
    ("options", "output_path"),
    [
        ([], Path("tests/fixtures/embeded_documents")),
        (["--output-json-path", "embedded"], Path("embedded")),
    ],
    ids=["default-output", "custom-output"],
)
def test_embedding_saves_returned_document(
    services: SimpleNamespace,
    dotenv_loader: Mock,
    options: list[str],
    output_path: Path,
) -> None:
    calls = Mock()
    calls.attach_mock(dotenv_loader, "load_dotenv")
    for name, service in vars(services).items():
        calls.attach_mock(service, name)

    assert cli.main(["embed-and-save-document", "source.json", *options]) == 0

    assert calls.mock_calls == [
        call.load_dotenv(),
        call.embed_document(Path("source.json")),
        call.save_embeded_document_json(
            services.embed_document.return_value, output_path
        ),
    ]


@pytest.mark.parametrize(
    ("options", "db_path", "artifacts_dir"),
    [
        ([], Path("configured-db"), Path("tests/fixtures/data/ingestion")),
        (
            ["--db-path", "custom-db", "--artifacts-dir", "custom-artifacts"],
            Path("custom-db"),
            Path("custom-artifacts"),
        ),
    ],
    ids=["default-paths", "custom-paths"],
)
def test_ingest_to_db_indexes_embedded_document(
    options: list[str],
    db_path: Path,
    artifacts_dir: Path,
    services: SimpleNamespace,
    dotenv_loader: Mock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = SimpleNamespace(db_path=Path("configured-db"))
    monkeypatch.setattr(cli.RetrievalSettings, "from_env", Mock(return_value=settings))

    pdf_sha256 = "a" * 64
    prepared_pdf = Path("prepared.pdf")
    converted_json = Path("converted.json")
    captioned_json = Path("captioned.json")
    document = SimpleNamespace(document_id="source-id")
    services.calculate_sha256.return_value = pdf_sha256
    services.fix_pdf_orientation.return_value = prepared_pdf
    services.convert_document.return_value = converted_json
    services.associate_table_captions.return_value = captioned_json
    services.embed_document.return_value = document
    services.upsert_document.return_value = 3

    client_context = MagicMock()
    client = Mock()
    collection = client.get_or_create_collection.return_value
    client_context.__enter__.return_value = client
    services.persistent_client.return_value = client_context

    calls = Mock()
    calls.attach_mock(dotenv_loader, "load_dotenv")
    for name, service in vars(services).items():
        calls.attach_mock(service, name)
    calls.attach_mock(client.get_or_create_collection, "get_or_create_collection")

    assert cli.main(["ingest-to-db", "source.pdf", "evidence", *options]) == 0

    assert calls.method_calls == [
        call.load_dotenv(),
        call.calculate_sha256(Path("source.pdf")),
        call.fix_pdf_orientation(
            Path("source.pdf"), artifacts_dir / f"{pdf_sha256}_prepared.pdf"
        ),
        call.convert_document(prepared_pdf, artifacts_dir),
        call.associate_table_captions(converted_json),
        call.embed_document(captioned_json, pdf_sha256),
        call.persistent_client(path=str(db_path)),
        call.get_or_create_collection(
            name="evidence",
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        ),
        call.upsert_document(collection, document),
    ]
    assert capsys.readouterr().out == "Indexed 3 chunks from source-id in evidence.\n"
