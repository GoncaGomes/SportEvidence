from dataclasses import dataclass
from math import atan2, degrees, isfinite
from pathlib import Path

import pymupdf

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TextLine:
    text: str
    bbox: BBox
    angle: int | None


@dataclass(frozen=True)
class PageMapping:
    prepared_page: int
    source_page: int
    source_bbox: BBox
    rotation_clockwise: int


@dataclass(frozen=True)
class TextRegion:
    angle: int | None
    bbox: BBox
    lines: tuple[TextLine, ...]


def extract_text_lines(
    page: pymupdf.Page,
    angle_tolerance: float = 5.0,
) -> list[TextLine]:
    if not 0 <= angle_tolerance < 45:
        raise ValueError("angle_tolerance must satisfy 0 <= value < 45")

    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    page_data = page.get_text("dict", flags=flags)
    lines = []

    for block in page_data["blocks"]:
        if block["type"] != 0:
            continue

        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"])
            if not text.strip():
                continue

            dx, dy = line["dir"]
            raw_angle = (-degrees(atan2(dy, dx))) % 360

            nearest_angle = int(round(raw_angle / 90) * 90) % 360
            difference = abs((raw_angle - nearest_angle + 180) % 360 - 180)
            angle = nearest_angle if difference <= angle_tolerance else None

            x0, y0, x1, y1 = line["bbox"]
            lines.append(
                TextLine(
                    text=text,
                    bbox=(x0, y0, x1, y1),
                    angle=angle,
                )
            )

    return lines


def merge_oriented_lines(
    lines: list[TextLine],
    angle: int,
) -> TextRegion | None:
    """Combine text lines with the requested orientation into one region."""
    if angle not in (0, 90, 180, 270):
        raise ValueError("angle must be 0, 90, 180 or 270")

    selected = [line for line in lines if line.angle == angle]

    if not selected:
        return None

    bbox = pymupdf.Rect(selected[0].bbox)

    for line in selected[1:]:
        bbox |= pymupdf.Rect(line.bbox)

    return TextRegion(
        angle=angle,
        bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
        lines=tuple(selected),
    )


def append_region_page(
    source: pymupdf.Document,
    output: pymupdf.Document,
    page_index: int,
    region: TextRegion,
    padding: float = 3.0,
    page_margin: float = 36.0,
) -> PageMapping:
    """Append an upright region and return its source mapping."""
    if region.angle not in (0, 90, 180, 270):
        raise ValueError("Region must have a known cardinal orientation")

    if not isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and non-negative")

    if not isfinite(page_margin) or page_margin < 0:
        raise ValueError("page_margin must be finite and non-negative")

    page = source[page_index]

    if page.rotation != 0:
        raise ValueError("Normalize page rotation before extracting regions")

    x0, y0, x1, y1 = region.bbox
    clip = (
        pymupdf.Rect(
            x0 - padding,
            y0 - padding,
            x1 + padding,
            y1 + padding,
        )
        & page.rect
    )

    if clip.is_empty or clip.is_infinite:
        raise ValueError("Region must intersect the source page")

    content_width = clip.width
    content_height = clip.height

    page_width = page.rect.width
    page_height = page.rect.height

    if region.angle in (90, 270):
        content_width, content_height = (
            content_height,
            content_width,
        )
        page_width, page_height = page_height, page_width

    page_width = max(
        page_width,
        content_width + 2 * page_margin,
    )
    page_height = max(
        page_height,
        content_height + 2 * page_margin,
    )

    left = (page_width - content_width) / 2

    placement = pymupdf.Rect(
        left,
        page_margin,
        left + content_width,
        page_margin + content_height,
    )

    with pymupdf.open() as cropped:
        cropped.insert_pdf(
            source,
            from_page=page_index,
            to_page=page_index,
        )

        cropped[0].clip_to_rect(clip)

        target = output.new_page(
            width=page_width,
            height=page_height,
        )

        target.show_pdf_page(
            placement,
            cropped,
            0,
            clip=clip,
            rotate=-region.angle,
        )

    return PageMapping(
        prepared_page=target.number + 1,
        source_page=page_index + 1,
        source_bbox=tuple(clip),
        rotation_clockwise=region.angle,
    )


def append_remaining_page(
    source: pymupdf.Document,
    output: pymupdf.Document,
    page_index: int,
    excluded_bbox: BBox,
) -> PageMapping:
    """Append the source page with the exported area removed."""
    with pymupdf.open() as remaining:
        remaining.insert_pdf(
            source,
            from_page=page_index,
            to_page=page_index,
        )

        page = remaining[0]

        page.add_redact_annot(
            pymupdf.Rect(excluded_bbox),
            fill=False,
            cross_out=False,
        )

        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
            graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
            text=pymupdf.PDF_REDACT_TEXT_REMOVE,
        )

        output.insert_pdf(remaining)

    return PageMapping(
        prepared_page=output.page_count,
        source_page=page_index + 1,
        source_bbox=tuple(source[page_index].rect),
        rotation_clockwise=0,
    )


def fix_pdf_orientation(
    pdf_path: Path,
    output_path: Path,
    min_rotated_chars: int = 100,
) -> list[PageMapping]:
    """Prepare rotated content and save the full PDF."""
    if pdf_path.resolve() == output_path.resolve():
        raise ValueError("Source and output paths must differ")

    if min_rotated_chars < 1:
        raise ValueError("min_rotated_chars must be positive")

    mappings: list[PageMapping] = []

    with (
        pymupdf.open(pdf_path) as source,
        pymupdf.open() as output,
    ):
        for page_index, page in enumerate(source):
            lines = extract_text_lines(page)
            rotated_region = merge_oriented_lines(
                lines,
                angle=90,
            )

            rotated_chars = (
                0
                if rotated_region is None
                else sum(
                    not character.isspace()
                    for line in rotated_region.lines
                    for character in line.text
                )
            )

            if rotated_region is None or rotated_chars < min_rotated_chars:
                output.insert_pdf(
                    source,
                    from_page=page_index,
                    to_page=page_index,
                )

                mappings.append(
                    PageMapping(
                        prepared_page=output.page_count,
                        source_page=page_index + 1,
                        source_bbox=tuple(page.rect),
                        rotation_clockwise=0,
                    )
                )
                continue

            table_mapping = append_region_page(
                source,
                output,
                page_index,
                rotated_region,
            )

            excluded = pymupdf.Rect(table_mapping.source_bbox)

            if any(
                line.angle != 90 and excluded.intersects(pymupdf.Rect(line.bbox))
                for line in lines
            ):
                raise ValueError(
                    f"Rotated crop overlaps other text on source page {page_index + 1}"
                )

            remaining_mapping = append_remaining_page(
                source,
                output,
                page_index,
                excluded_bbox=table_mapping.source_bbox,
            )

            mappings.extend([table_mapping, remaining_mapping])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(
            output_path,
            garbage=4,
            deflate=True,
        )

    return mappings

