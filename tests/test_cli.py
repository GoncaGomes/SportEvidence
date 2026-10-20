from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sportevidence import cli
from sportevidence.models import RetrievedChunk, RetrievedQueries


@pytest.fixture(autouse=True)
def dotenv_loader(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Do not read local configuration or alter the environment during CLI tests."""
    loader = Mock(return_value=False)
    monkeypatch.setattr(cli, "load_dotenv", loader)
    return loader


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Keep CLI tests independent of PDF, network, and database services."""
    mocks = SimpleNamespace(
        chunk_document=Mock(),
        embed_chunks=Mock(),
        upsert_document=Mock(),
        query_collection=Mock(),
    )
    for name, service in vars(mocks).items():
        monkeypatch.setattr(cli, name, service)
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
    assert "query-db" in capsys.readouterr().out
    dotenv_loader.assert_not_called()


def test_command_is_required() -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main([])
    assert raised.value.code == 2


def test_ingestion_passes_arguments_and_returns_success(
    services: SimpleNamespace,
    dotenv_loader: Mock,
) -> None:
    calls = Mock()
    calls.attach_mock(dotenv_loader, "load_dotenv")
    calls.attach_mock(services.chunk_document, "chunk_document")
    assert (
        cli.main(
            [
                "document-to-db",
                "source.pdf",
                "--input-db-path",
                "database",
                "--collection-name",
                "evidence",
            ]
        )
        == 0
    )
    services.chunk_document.assert_called_once_with(Path("source.pdf"))
    services.embed_chunks.assert_called_once_with(services.chunk_document.return_value)
    services.upsert_document.assert_called_once_with(
        services.embed_chunks.return_value, Path("database"), collection_name="evidence"
    )
    assert calls.mock_calls[0] == ("load_dotenv", (), {})
    assert calls.mock_calls[1] == ("chunk_document", (Path("source.pdf"),), {})


@pytest.mark.parametrize("queries", [["Recovery"], ["Recovery", "Nutrition"]])
def test_query_prints_all_groups(
    services: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    queries: list[str],
) -> None:
    services.query_collection.return_value = RetrievedQueries(
        queries,
        [
            [RetrievedChunk("a#0001", "Evidence text", [1.0], "a", "a.pdf", 0.25)]
            for _ in queries
        ],
    )
    arguments = ["query-db", "database", "evidence", "--n-results", "3"]
    for query in queries:
        arguments.extend(["--query", query])

    assert cli.main(arguments) == 0

    services.query_collection.assert_called_once_with(
        query=queries, db_path=Path("database"), collection_name="evidence", n_results=3
    )
    output = capsys.readouterr().out
    for query in queries:
        assert f"Query: {query}" in output
    assert output.count("Text: Evidence text") == len(queries)
    assert "Source: a.pdf" in output
    assert "Distance: 0.250000" in output


def test_query_prints_empty_results(
    services: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    services.query_collection.return_value = RetrievedQueries(["Recovery"], [[]])
    assert cli.main(["query-db", "database", "evidence", "--query", "Recovery"]) == 0
    assert "No results." in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["--query", "Recovery", "--n-results", "0"],
        ["--query", "Recovery", "--n-results", "-1"],
        ["--query", "Recovery", "--n-results", "invalid"],
        ["--query", " "],
    ],
)
def test_invalid_query_arguments_skip_retrieval(
    services: SimpleNamespace, arguments: list[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["query-db", "database", "evidence", *arguments])
    assert raised.value.code == 2
    services.query_collection.assert_not_called()
