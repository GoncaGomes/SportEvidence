from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData

from sportevidence.ingestion.table_captions import associate_table_captions


def add_table(
    document: DoclingDocument, caption: str, headers: tuple[str, ...]
) -> None:
    caption_item = (
        document.add_text(label=DocItemLabel.CAPTION, text=caption) if caption else None
    )
    document.add_table(
        caption=caption_item,
        data=TableData(
            num_rows=1,
            num_cols=len(headers),
            table_cells=[
                TableCell(
                    text=text,
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=index,
                    end_col_offset_idx=index + 1,
                )
                for index, text in enumerate(headers)
            ],
        ),
    )


def associate(document: DoclingDocument, tmp_path: Path) -> DoclingDocument:
    source = tmp_path / "tables.json"
    document.save_as_json(source)
    associate_table_captions(source)
    return DoclingDocument.load_from_json(tmp_path / "tables_associated.json")


@pytest.mark.parametrize("gap", [0, 12])
def test_same_identifier_preserves_original_despite_header_and_distance_changes(
    tmp_path: Path, gap: int
) -> None:
    document = DoclingDocument(name="tables")
    add_table(document, "Table 2 Original caption", ("Study", "Strength training"))
    for _ in range(gap):
        document.add_text(label=DocItemLabel.TEXT, text="Intervening text")
    add_table(document, "Table 2 (continued)", ("Study", "Strength", "training"))
    add_table(document, "Table 2 (continued)", ("Different header",))

    result = associate(document, tmp_path)

    assert all(table.captions == result.tables[0].captions for table in result.tables)


def test_unmatched_fragment_does_not_erase_original_caption(tmp_path: Path) -> None:
    document = DoclingDocument(name="tables")
    add_table(document, "Table 3 Original caption", ("Study", "G"))
    add_table(document, "", ("Table 3 Study", "(continued) G"))
    add_table(document, "Table 3 (continued)", ("Study", "G"))

    result = associate(document, tmp_path)

    assert result.tables[1].captions == []
    assert result.tables[2].captions == result.tables[0].captions


def test_different_identifiers_keep_their_own_captions(tmp_path: Path) -> None:
    document = DoclingDocument(name="tables")
    add_table(document, "Table 2 Original caption", ("Study",))
    add_table(document, "Table 3 Another caption", ("Study",))

    result = associate(document, tmp_path)

    assert result.tables[0].captions != result.tables[1].captions
    assert (
        result.tables[1].captions[0].resolve(result).text == "Table 3 Another caption"
    )


@pytest.mark.parametrize(
    ("headers", "gap", "expected_associated"),
    [
        ((" study ", "G"), 0, True),
        (("Different",), 0, False),
        (("Study", "G"), 12, False),
    ],
)
def test_captionless_fragments_require_nearby_matching_headers(
    tmp_path: Path, headers: tuple[str, ...], gap: int, expected_associated: bool
) -> None:
    document = DoclingDocument(name="tables")
    add_table(document, "Table 2 Original caption", ("Study", "G"))
    for _ in range(gap):
        document.add_text(label=DocItemLabel.TEXT, text="Intervening text")
    add_table(document, "", headers)

    result = associate(document, tmp_path)

    assert bool(result.tables[1].captions) == expected_associated
    if expected_associated:
        assert result.tables[1].captions == result.tables[0].captions
