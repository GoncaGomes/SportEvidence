from pathlib import Path
from unittest.mock import Mock

import pytest

from sportevidence.ingestion import parsing


@pytest.mark.parametrize("suffix", [".pdf", ".PDF"])
def test_convert_document_saves_json_and_returns_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    pdf_path = tmp_path / f"paper{suffix}"
    pdf_path.touch()
    fake_converter = Mock()
    fake_document = Mock()
    fake_result = Mock()
    fake_converter.convert.return_value = fake_result
    fake_result.document = fake_document

    monkeypatch.setattr(parsing, "converter", fake_converter)

    expected_path = Path("tests/fixtures/corpus_json/paper.json")

    result = parsing.convert_document(pdf_path)

    fake_converter.convert.assert_called_once_with(pdf_path)
    fake_document.save_as_json.assert_called_once_with(expected_path)

    assert result == expected_path


@pytest.mark.parametrize(
    "input_kind , expected_error",
    [
        ("missing", FileNotFoundError),
        ("wrong_extension", ValueError),
        ("directory", IsADirectoryError),
    ],
)
def test_invalid_input_never_calls_the_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_kind: str,
    expected_error: type[Exception],
) -> None:
    if input_kind == "missing":
        input_path = tmp_path / "missing.pdf"
    elif input_kind == "wrong_extension":
        input_path = tmp_path / "wrong_extension.txt"
        input_path.touch()
    else:
        input_path = tmp_path / "directory"
        input_path.mkdir()

    fake_converter = Mock()
    monkeypatch.setattr(parsing, "converter", fake_converter)

    with pytest.raises(expected_error):
        parsing.convert_document(input_path)

    fake_converter.convert.assert_not_called()


def test_conversion_errors_propagates_without_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "paper.PDF"
    pdf_path.touch()
    fake_converter = Mock()
    fake_document = Mock()
    fake_result = Mock()
    fake_result.document = fake_document
    fake_converter.convert.return_value = fake_result
    fake_converter.convert.side_effect = ValueError("Conversion failed")

    monkeypatch.setattr(parsing, "converter", fake_converter)

    with pytest.raises(ValueError, match="Conversion failed"):
        parsing.convert_document(pdf_path)

    fake_converter.convert.assert_called_once_with(pdf_path)
    fake_document.save_as_json.assert_not_called()


def test_save_json_error_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "paper.PDF"
    pdf_path.touch()
    fake_converter = Mock()
    fake_document = Mock()
    fake_result = Mock()
    fake_converter.convert.return_value = fake_result
    fake_result.document = fake_document
    fake_result.document.save_as_json.side_effect = OSError("Save failed")

    monkeypatch.setattr(parsing, "converter", fake_converter)
    expected_path = Path("tests/fixtures/corpus_json/paper.json")

    with pytest.raises(OSError, match="Save failed"):
        parsing.convert_document(pdf_path)

    fake_converter.convert.assert_called_once_with(pdf_path)
    fake_document.save_as_json.assert_called_once_with(expected_path)
