import re
from pathlib import Path

from docling_core.types.doc import DoclingDocument, TableItem


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def extract_table_identifier(caption: str) -> int | None:
    pattern = r"^\s*(?:table|tab\.?)\s*(\d+)\b"

    match = re.match(pattern, caption, flags=re.IGNORECASE)

    if match is None:
        return None

    return int(match.group(1))


def extract_first_row(table: TableItem) -> tuple[str, ...]:
    cells = [cell for cell in table.data.table_cells if cell.start_row_offset_idx == 0]
    ordered_cells = sorted(
        cells,
        key=lambda cell: cell.start_col_offset_idx,
    )

    return tuple(normalize_text(cell.text) for cell in ordered_cells)


def have_matching_headers(
    previous: TableItem,
    current: TableItem,
) -> bool:
    previous_header = extract_first_row(previous)
    current_header = extract_first_row(current)

    if not any(previous_header) or not any(current_header):
        return False

    return (
        previous.data.num_cols == current.data.num_cols
        and previous_header == current_header
    )


def associate_table_captions(json_path: Path) -> Path:
    document = DoclingDocument.load_from_json(json_path)

    previous_table: TableItem | None = None
    previous_identifier: int | None = None
    previous_position: int | None = None
    original_tables: dict[int, TableItem] = {}

    max_intervening_items = 10

    for position, (item, _) in enumerate(document.iterate_items()):
        if not isinstance(item, TableItem):
            continue
        captions = [reference.resolve(document).text for reference in item.captions]
        caption = " ".join(captions).strip()
        identifier = extract_table_identifier(caption)

        logical_identifier = identifier
        caption_source = (
            original_tables.get(identifier) if identifier is not None else None
        )

        if (
            caption_source is None
            and not caption
            and previous_table is not None
            and previous_position is not None
            and previous_identifier is not None
        ):
            intervening_items = position - previous_position - 1

            if intervening_items <= max_intervening_items and have_matching_headers(
                previous_table, item
            ):
                caption_source = original_tables.get(previous_identifier)
                if caption_source is not None:
                    logical_identifier = previous_identifier

        associated = caption_source is not None
        if associated:
            " ".join(
                reference.resolve(document).text
                for reference in caption_source.captions
            ).strip()
            item.captions = caption_source.captions.copy()
        elif identifier is not None:
            original_tables[identifier] = item
        else:
            pass

        previous_table = item
        previous_identifier = logical_identifier
        previous_position = position

    output_path = json_path.with_name(f"{json_path.stem}_associated.json")
    document.save_as_json(output_path)

    return output_path
