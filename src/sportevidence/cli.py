from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.client import Client
from dotenv import load_dotenv

from sportevidence.config import RetrievalSettings
from sportevidence.ingestion.embeddings import (
    embed_document,
    save_embeded_document_json,
)
from sportevidence.ingestion.orientation import fix_pdf_orientation
from sportevidence.ingestion.parsing import convert_document
from sportevidence.ingestion.table_captions import associate_table_captions
from sportevidence.storage.vector_store import upsert_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sportevidence",
        description="Ingest and search sports-science evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    RetrievalSettings.from_env()

    fix_pdf_orientation = subparsers.add_parser(
        "fix-pdf-orientation", help="Fix the orientation of a PDF document"
    )
    fix_pdf_orientation.add_argument(
        "input_pdf_path", type=Path, help="Path to the input PDF file"
    )
    fix_pdf_orientation.add_argument(
        "output_pdf_path", type=Path, help="Output path for the fixed PDF file"
    )

    convert_pdf_docling = subparsers.add_parser(
        "convert-pdf-docling", help="Convert a PDF to Doclingo format"
    )

    convert_pdf_docling.add_argument(
        "input_pdf_path", type=Path, help="Input path to the fixed PDF file"
    )

    convert_pdf_docling.add_argument(
        "--output-json-path",
        type=Path,
        default=Path("tests/fixtures/corpus_json"),
        help="Output path for the Doclingo file",
    )

    associate_captions = subparsers.add_parser(
        "associate-captions", help="Associate captions with chunks"
    )

    associate_captions.add_argument(
        "input_json_path", type=Path, help="Path to docling json file"
    )

    embed_and_save_document = subparsers.add_parser(
        "embed-and-save-document", help="Embed and save a document in json format"
    )

    embed_and_save_document.add_argument(
        "json_path",
        type=Path,
        help="Input path to the caption associated json file",
    )

    embed_and_save_document.add_argument(
        "--output-json-path",
        type=Path,
        default=Path("tests/fixtures/embeded_documents"),
        help="Output path for the Doclingo file",
    )

    ingest_to_db = subparsers.add_parser(
        "ingest-to-db", help="Inegst a pdf file and insert it in a collection"
    )
    ingest_to_db.add_argument(
        "input_pdf_path", type=Path, help="Path to the input PDF file"
    )
    ingest_to_db.add_argument(
        "collection_name", type=str, help="Name of the collection"
    )
    ingest_to_db.add_argument(
        "--db-path",
        type=Path,
        help="Path to the database",
    )
    ingest_to_db.add_argument(
        "--artifacts-dir", type=Path, default=Path("tests/fixtures/data/ingestion")
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv()

    if args.command == "fix-pdf-orientation":
        fix_pdf_orientation(args.input_pdf_path, args.output_pdf_path)
        return 0

    if args.command == "convert-pdf-docling":
        convert_document(args.input_pdf_path, args.output_json_path)
        return 0

    if args.command == "associate-captions":
        associate_table_captions(args.input_json_path)
        return 0

    if args.command == "embed-and-save-document":
        doc = embed_document(args.json_path)
        save_embeded_document_json(doc, args.output_json_path)
        return 0

    if args.command == "ingest-to-db":
        artifact_dir = args.artifacts_dir
        prepared_pdf_path = fix_pdf_orientation(
            args.input_pdf_path,
            artifact_dir / f"{args.input_pdf_path.stem}_prepared.pdf",
        )
        docling_json = convert_document(prepared_pdf_path, artifact_dir)
        table_associate_json_path = associate_table_captions(docling_json)
        doc = embed_document(table_associate_json_path)

        settings = RetrievalSettings.from_env()
        db_path = args.db_path or settings.db_path
        persistent_client = cast(Client, chromadb.PersistentClient(path=str(db_path)))
        with persistent_client as client:
            collection = client.get_or_create_collection(
                name=args.collection_name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
            )
            written = upsert_document(collection, doc)

        print(
            f"Indexed {written} chunks from {doc.document_id} "
            f"in {args.collection_name}."
        )
        return 0
    parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
