"""A query interface with application configuration bound outside tool inputs."""

from typing import TypedDict

from sportevidence.config import RetrievalSettings
from sportevidence.storage.vector_store import query_collection


class Passage(TypedDict):
    chunk_id: str
    text: str
    document_id: str
    source_path: str
    distance: float


class QueryResult(TypedDict):
    query: str
    passages: list[Passage]


class QueryTool:
    """Bind database settings once; expose only query text and result count."""

    def __init__(self, settings: RetrievalSettings | None = None) -> None:
        self.settings = (
            settings if settings is not None else RetrievalSettings.from_env()
        )

    def query(self, query: str, n_results: int = 5) -> QueryResult:
        """Retrieve source passages as JSON-serializable data without vectors."""
        result = query_collection(
            query=[query],
            db_path=self.settings.db_path,
            collection_name=self.settings.collection_name,
            n_results=n_results,
        )
        return QueryResult(
            query=query,
            passages=[
                Passage(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    document_id=chunk.document_id,
                    source_path=chunk.source_path,
                    distance=chunk.distance,
                )
                for chunk in result.retrieved_chunks[0]
            ],
        )
