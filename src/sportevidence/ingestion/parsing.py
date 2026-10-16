from pathlib import Path
from markitdown import MarkItDown


CONVERTED_PATH = Path("corpus/converted/test.md")
INPUT_PATH = Path("corpus/fixtures/40279_2023_Article_1978.pdf")
md = MarkItDown()


def parse_markdown(path: Path) -> str:
    return md.convert(str(path))


if __name__ == "__main__":
    result = parse_markdown(INPUT_PATH)
    CONVERTED_PATH.write_text(result.markdown, encoding="utf-8")

    