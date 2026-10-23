import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from sportevidence import query
from sportevidence.config import RetrievalSettings
from sportevidence.models import RetrievedChunk, RetrievedQueries


def test_query_tool_returns_serializable_passages_without_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieve = Mock(
        return_value=RetrievedQueries(
            queries=["Recovery"],
            retrieved_chunks=[
                [RetrievedChunk("a#0001", "Evidence", [1.0], "a", "a.pdf", 0.1)]
            ],
        )
    )
    monkeypatch.setattr(query, "query_collection", retrieve)
    tool = query.QueryTool(RetrievalSettings(Path("database"), "evidence"))

    result = tool.query("Recovery", n_results=3)

    retrieve.assert_called_once_with(
        query=["Recovery"],
        db_path=Path("database"),
        collection_name="evidence",
        n_results=3,
    )
    assert json.loads(json.dumps(result)) == {
        "query": "Recovery",
        "passages": [
            {
                "chunk_id": "a#0001",
                "text": "Evidence",
                "document_id": "a",
                "source_path": "a.pdf",
                "distance": 0.1,
            }
        ],
    }


def test_query_tool_returns_empty_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        query,
        "query_collection",
        Mock(return_value=RetrievedQueries(["Recovery"], [[]])),
    )

    assert query.QueryTool().query("Recovery") == {
        "query": "Recovery",
        "passages": [],
    }
