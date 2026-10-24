from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from sportevidence.config import RetrievalSettings
from sportevidence.ingestion.chunking import chunk_document
from sportevidence.ingestion.embeddings import embed_chunks
from sportevidence.storage.vector_store import query_collection, upsert_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sportevidence",
        description="Ingest and search sports-science evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    defaults = RetrievalSettings.from_env()

    document_to_db = subparsers.add_parser(
        "document-to-db", help="Add a PDF document to the database"
    )
    document_to_db.add_argument(
        "input_pdf_path", type=Path, help="Path to the input PDF file"
    )
    document_to_db.add_argument(
        "--input-db-path",
        type=Path,
        default=defaults.db_path,
        help="Path to the persistent database (default: %(default)s)",
    )
    document_to_db.add_argument(
        "--collection-name",
        default=defaults.collection_name,
        help="Collection to store the document in (default: %(default)s)",
    )

    query_db = subparsers.add_parser("query-db", help="Query the database")
    query_db.add_argument("database_path", type=Path, help="Path to the database")
    query_db.add_argument("collection_name", help="Collection to search")
    query_db.add_argument(
        "--query", action="append", required=True, help="Query text; repeat for a batch"
    )
    query_db.add_argument(
        "--n-results", type=int, default=5, help="Number of results per query"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv()

    if args.command == "document-to-db":
        document = chunk_document(args.input_pdf_path)
        upsert_document(
            embed_chunks(document),
            args.input_db_path,
            collection_name=args.collection_name,
        )
        return 0

    if args.command == "query-db":
        if args.n_results <= 0:
            parser.error("--n-results must be positive")
        if any(not query.strip() for query in args.query):
            parser.error("--query must not be empty or whitespace-only")

        result = query_collection(
            query=args.query,
            db_path=args.database_path,
            collection_name=args.collection_name,
            n_results=args.n_results,
        )
        for query, chunks in zip(result.queries, result.retrieved_chunks, strict=True):
            print(f"Query: {query}")
            if not chunks:
                print("  No results.")
            for chunk in chunks:
                print(f"  Chunk_id: {chunk.chunk_id}")
                print(f"  Document_id: {chunk.document_id}")
                print(f"  Source: {chunk.source_path}")
                print(f"  Distance: {chunk.distance:.6f}")
                print(f"  Text: {chunk.text}\n")
                print(f"  Embeddings: {chunk.embedding[:3]}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
