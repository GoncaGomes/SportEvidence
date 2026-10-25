from pathlib import Path

import pymupdf
import pytest

from sportevidence.ingestion import orientation


@pytest.fixture
def source_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "source.pdf"
    with pymupdf.open() as source:
        page = source.new_page(width=600, height=800)
        page.insert_text((72, 72), "Page heading")
        page.insert_text((150, 350), "Rotated table column one", rotate=90)
        page.insert_text((240, 350), "Rotated table column two", rotate=90)
        source.save(pdf_path)
    return pdf_path


@pytest.mark.parametrize("threshold, expected_pages", [(1, 2), (100, 1)])
def test_prepare_pdf_preserves_text_and_rotation_threshold(
    source_pdf: Path,
    tmp_path: Path,
    threshold: int,
    expected_pages: int,
) -> None:
    output_path = tmp_path / "prepared" / "output.pdf"

    mappings = orientation.prepare_pdf(
        source_pdf, output_path, min_rotated_chars=threshold
    )

    with pymupdf.open(source_pdf) as source, pymupdf.open(output_path) as prepared:
        assert prepared.page_count == expected_pages
        assert sorted(
            line for page in prepared for line in page.get_text().splitlines()
        ) == sorted(source[0].get_text().splitlines())
        assert [mapping.prepared_page for mapping in mappings] == list(
            range(1, expected_pages + 1)
        )
        assert all(mapping.source_page == 1 for mapping in mappings)
        if expected_pages == 2:
            table_lines = orientation.extract_text_lines(prepared[0])
            assert {line.text for line in table_lines} == {
                "Rotated table column one",
                "Rotated table column two",
            }
            assert all(line.angle == 0 for line in table_lines)
            assert prepared[1].get_text().strip() == "Page heading"
        else:
            assert any(
                line.angle == 90 for line in orientation.extract_text_lines(prepared[0])
            )


@pytest.mark.parametrize("existing_output", [False, True])
def test_prepare_pdf_does_not_write_output_when_rotated_crop_overlaps_text(
    source_pdf: Path,
    tmp_path: Path,
    existing_output: bool,
) -> None:
    output_path = tmp_path / "prepared" / "output.pdf"
    original_content = b"Existing output must remain untouched"
    if existing_output:
        output_path.parent.mkdir()
        output_path.write_bytes(original_content)

    with pymupdf.open(source_pdf) as source:
        source[0].insert_text((160, 280), "Overlapping text")
        source.saveIncr()

    with pytest.raises(ValueError, match="Rotated crop overlaps other text"):
        orientation.prepare_pdf(source_pdf, output_path, min_rotated_chars=1)

    if existing_output:
        assert output_path.read_bytes() == original_content
    else:
        assert not output_path.parent.exists()
