from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

INPUT_DOC_PATH = Path("tests/fixtures/40279_2023_Article_1978.pdf")
accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.AUTO)
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.generate_parsed_pages = True
pipeline_options.accelerator_options = accelerator_options
pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    },
    show_progress=True,
)


def _validate_corpus_file(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {path}")

    return path


def convert_document(path: Path) -> Path:
    _validate_corpus_file(path)
    result = converter.convert(Path(path))
    json_path = Path(f"tests/fixtures/corpus_json/{path.stem}.json")
    result.document.save_as_json(json_path)
    return json_path
