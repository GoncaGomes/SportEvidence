from pathlib import Path

import pymupdf
import pytest

from sportevidence.ingestion import parsing
from sportevidence.ingestion.parsing import parse_markdown


def test_parse_markdown_rejects_missing_file(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        parse_markdown(missing_pdf)


def test_parse_markdown_rejects_wrong_extension(tmp_path: Path) -> None:
    wrong_extension = tmp_path / "wrong_extension.txt"
    wrong_extension.write_text("Something", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_markdown(wrong_extension)


def test_parse_markdown_rejects_folders(tmp_path: Path) -> None:
    directory_path = tmp_path / "directory"
    directory_path.mkdir()
    with pytest.raises(IsADirectoryError):
        parse_markdown(directory_path)


def test_parse_markdown_rejects_empty_file(tmp_path: Path) -> None:
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(pymupdf.EmptyFileError):
        parse_markdown(empty_pdf)


def test_parse_markdown_returns_converter_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.touch()

    expected_markdown = "# Example \n 1 Intro \n 2 SoA"

    def fake_to_markdown(path: Path) -> str:
        return expected_markdown

    monkeypatch.setattr(parsing.pymupdf4llm, "to_markdown", fake_to_markdown)

    result = parsing.parse_markdown(pdf_path)

    assert result == expected_markdown


def test_parse_markdown_converts_real_pdf(tmp_path: Path) -> None:
    pdf_patch = tmp_path / "document.pdf"
    expected_text = "SportEvidence integration test."

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), expected_text)
        document.save(str(pdf_patch))

    result = parsing.parse_markdown(pdf_patch)

    assert isinstance(result, str)
    assert expected_text in result


def test_empty_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_file = tmp_path / "document.pdf"
    pdf_file.touch()
    expected_markdown = ""

    def fake_to_markdown(path: Path) -> str:
        return expected_markdown

    monkeypatch.setattr(parsing.pymupdf4llm, "to_markdown", fake_to_markdown)

    with pytest.raises(ValueError):
        parse_markdown(pdf_file)


def test_caps_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_file = tmp_path / "document.PDF"
    pdf_file.touch()
    expected_markdown = "Something"

    def fake_to_markdown(path: Path) -> str:
        return expected_markdown

    monkeypatch.setattr(parsing.pymupdf4llm, "to_markdown", fake_to_markdown)

    result = parse_markdown(pdf_file)
    assert result == expected_markdown
