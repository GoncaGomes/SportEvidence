import hashlib
from pathlib import Path

import pytest

from sportevidence.ingestion import chunking
from sportevidence.models import Chunk, Document


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_chunk_text_returns_no_chunks_for_blank_text(text: str) -> None:
    assert chunking.chunk_text(text, "abc123") == []


def test_chunk_text_preserves_short_text() -> None:
    result = chunking.chunk_text("Short text.", "abc123")

    assert result == [Chunk("abc123#0001", "Short text.")]
    assert result[0].embedding is None


@pytest.mark.parametrize(
    ("overlap", "expected_texts"),
    [(0, ["abcde", "fghij", "kl"]), (2, ["abcde", "defgh", "ghijk", "jkl"])],
)
def test_chunk_text_splits_with_expected_overlap(
    overlap: int, expected_texts: list[str]
) -> None:
    result = chunking.chunk_text(
        "abcdefghijkl", "abc123", chunk_size=5, overlap=overlap
    )

    assert [chunk.text for chunk in result] == expected_texts
    assert all(len(chunk.text) <= 5 for chunk in result)
    assert all(chunk.embedding is None for chunk in result)
    assert [chunk.chunk_id for chunk in result] == [
        f"abc123#{index:04d}" for index in range(1, len(expected_texts) + 1)
    ]
    assert result == chunking.chunk_text(
        "abcdefghijkl", "abc123", chunk_size=5, overlap=overlap
    )


@pytest.mark.parametrize("chunk_size, overlap", [(5, -1), (5, 6), (0, 0), (-1, 0)])
def test_chunk_text_rejects_invalid_sizes(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunking.chunk_text("Example", "abc123", chunk_size, overlap)


def test_chunk_text_accepts_overlap_equal_to_chunk_size() -> None:
    result = chunking.chunk_text("abcdef", "abc123", chunk_size=3, overlap=3)

    assert [chunk.text for chunk in result] == ["abc", "bcd", "cde", "def"]


@pytest.mark.parametrize(
    "payload",
    [b"", b"abc", b"a" * (1024 * 1024 + 17)],
    ids=["empty", "short", "multiple-read-buffers"],
)
def test_calculate_sha256_hashes_file_bytes(tmp_path: Path, payload: bytes) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(payload)

    assert chunking.calculate_sha256(source_path) == hashlib.sha256(payload).hexdigest()


def test_calculate_sha256_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        chunking.calculate_sha256(tmp_path / "missing.pdf")


def test_chunk_document_assembles_ordered_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.pdf"
    payload = b"Example PDF bytes"
    source_path.write_bytes(payload)
    document_id = hashlib.sha256(payload).hexdigest()
    markdown = "a" * 2500 + "b" * 500

    def fake_parse_markdown(path: Path) -> str:
        assert path == source_path
        return markdown

    monkeypatch.setattr(chunking, "parse_markdown", fake_parse_markdown)

    result = chunking.chunk_document(source_path)

    assert isinstance(result, Document)
    assert result.document_id == document_id
    assert result.source_path == source_path
    assert isinstance(result.chunks, tuple)
    assert result.chunks == (
        Chunk(f"{document_id}#0001", "a" * 2500),
        Chunk(
            f"{document_id}#0002",
            "a" * 250 + "b" * 500,
        ),
    )


def test_chunk_document_propagates_parser_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"Example PDF bytes")
    error = ValueError("Unable to parse PDF")

    def fake_parse_markdown(path: Path) -> str:
        raise error

    monkeypatch.setattr(chunking, "parse_markdown", fake_parse_markdown)

    with pytest.raises(ValueError) as raised:
        chunking.chunk_document(source_path)

    assert raised.value is error


def test_chunk_ids_distinguish_documents_with_the_same_hash_prefix() -> None:
    first = chunking.chunk_text("Same text", "abc12" + "0" * 59)
    second = chunking.chunk_text("Same text", "abc12" + "1" * 59)

    assert first[0].chunk_id != second[0].chunk_id
