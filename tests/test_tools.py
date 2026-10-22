from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool

from sportevidence.agent import tools
from sportevidence.config import RetrievalSettings
from sportevidence.models import RetrievedChunk, RetrievedQueries


def _payload() -> RetrievedQueries:
    return RetrievedQueries(
        queries=["One"],
        retrieved_chunks=[
            [
                RetrievedChunk(
                    chunk_id="a#0001",
                    text="First",
                    embedding=[1.0, 0.0],
                    document_id="a",
                    source_path="a.pdf",
                    distance=0.0,
                )
            ]
        ],
    )


def _tool(settings: RetrievalSettings) -> BaseTool:
    return tools.create_query_tool(settings)


def test_create_query_tool_returns_search_evidence_tool() -> None:
    settings = RetrievalSettings(db_path=Path("db"), collection_name="coll")

    search_tool = tools.create_query_tool(settings)

    assert isinstance(search_tool, BaseTool)
    assert search_tool.name == "search_evidence"


def test_search_evidence_passes_bound_settings_and_strips_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_mock = Mock(return_value=_payload())
    monkeypatch.setattr(tools, "query_collection", query_mock)
    settings = RetrievalSettings(db_path=Path("db"), collection_name="coll")
    search_tool = _tool(settings)

    result = search_tool.invoke({"query": ["One"], "n_results": 2})

    query_mock.assert_called_once_with(
        query=["One"],
        db_path=Path("db"),
        collection_name="coll",
        n_results=2,
    )
    assert result == {
        "queries": ["One"],
        "retrieved_chunks": [
            [
                {
                    "chunk_id": "a#0001",
                    "text": "First",
                    "document_id": "a",
                    "source_path": "a.pdf",
                    "distance": 0.0,
                }
            ]
        ],
    }


def test_search_evidence_uses_default_n_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_mock = Mock(return_value=_payload())
    monkeypatch.setattr(tools, "query_collection", query_mock)
    settings = RetrievalSettings(db_path=Path("db"), collection_name="coll")
    search_tool = _tool(settings)

    search_tool.invoke({"query": ["One"]})

    query_mock.assert_called_once_with(
        query=["One"],
        db_path=Path("db"),
        collection_name="coll",
        n_results=5,
    )


def test_create_query_tool_falls_back_to_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_mock = Mock(return_value=_payload())
    monkeypatch.setattr(tools, "query_collection", query_mock)
    fixed = RetrievalSettings(db_path=Path("env-db"), collection_name="env-coll")
    monkeypatch.setattr(
        tools, "RetrievalSettings", Mock(from_env=Mock(return_value=fixed))
    )

    search_tool = tools.create_query_tool()
    search_tool.invoke({"query": ["One"]})

    query_mock.assert_called_once_with(
        query=["One"],
        db_path=Path("env-db"),
        collection_name="env-coll",
        n_results=5,
    )


def test_search_evidence_rejects_blank_queries() -> None:
    settings = RetrievalSettings(db_path=Path("db"), collection_name="coll")
    search_tool = _tool(settings)

    with pytest.raises(ValueError, match="nonblank query"):
        search_tool.invoke({"query": ["   "]})
    with pytest.raises(ValueError, match="nonblank query"):
        search_tool.invoke({"query": []})
