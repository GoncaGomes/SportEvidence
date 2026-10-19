from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sportevidence.ingestion.chunking import chunk_document
from sportevidence.ingestion.embeddings import embed_chunks
from sportevidence.storage.vector_store import upsert_document


# create a parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sportevidence",
        description="A cientific chatbot for sport evidence",
    )
    subparser = parser.add_subparsers(dest="command")

    document_to_db = subparser.add_parser(
        "document-to-db",
        help="Add a document to the database",
    )

    document_to_db.add_argument(
        "input_pdf_path",
        type=Path,
        help="Path to the input PDF file",
    )
    document_to_db.add_argument(
        "--input-db-path",
        type=Path,
        default=Path("tests/fixtures/data"),
        help="Path to the input database",
    )
    document_to_db.add_argument(
        "--collection-name",
        type=Path,
        default="default_collection",
        help="Path to the input database",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "document-to-db":
        input_pdf_path = args.input_pdf_path
        input_db_path = args.input_db_path
        collection_name = args.collection_name
        document = chunk_document(input_pdf_path)
        embedded_document = embed_chunks(document)

        upsert_document(
            embedded_document,
            input_db_path,
            collection_name=collection_name,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
