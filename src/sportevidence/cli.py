from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from sportevidence.config import RetrievalSettings
from sportevidence.ingestion.orientation import fix_pdf_orientation
from sportevidence.ingestion.parsing import convert_document
from sportevidence.ingestion.table_captions import associate_table_captions


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

    associate_captions = subparsers.add_parser(
        "associate-captions", help="Associate captions with chunks"
    )

    associate_captions.add_argument(
        "input_json_path", type=Path, help="Path to docling json file"
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
        convert_document(args.input_pdf_path)
        return 0

    if args.command == "associate-captions":
        associate_table_captions(args.input_json_path)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
