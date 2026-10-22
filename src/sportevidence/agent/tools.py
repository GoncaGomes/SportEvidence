from dataclasses import asdict
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool

from sportevidence.config import RetrievalSettings
from sportevidence.storage.vector_store import query_collection


def create_query_tool(settings: RetrievalSettings | None = None) -> BaseTool:
    settings = settings if settings is not None else RetrievalSettings.from_env()

    @tool
    def search_evidence(query: list[str], n_results: int = 5) -> dict:
        """Search sports-science papers using one or more queries.

        Args:
            query: search queries. Each query returns its own result group.
            n_results: Maximum number of passages returned per query
        """

        if not query or any(not text.strip() for text in query):
            raise ValueError("Prvide at least one nonblank query.")

        result = query_collection(
            query=query,
            db_path=settings.db_path,
            collection_name=settings.collection_name,
            n_results=n_results,
        )

        payload = asdict(result)

        for group in payload["retrieved_chunks"]:
            for chunk in group:
                chunk.pop("embedding", None)

        return payload

    return search_evidence


def main():
    search_tool = create_query_tool(
        RetrievalSettings(
            db_path=Path("tests/fixtures/data"),
            collection_name="test_collection",
        )
    )
    result = search_tool.invoke(
        {
            "query": [
                "resistance training running economy",
                "strength training endurance running performance",
            ],
            "n_results": 1,
        }
    )
    print(result)


if __name__ == "__main__":
    raise SystemExit(main())
