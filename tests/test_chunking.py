from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData

from sportevidence.ingestion import chunking


def test_extract_title_returns_stripped_title_and_prefers_title_over_header() -> None:
    document = DoclingDocument(name="example")
    document.add_heading(text="Introduction", level=1)
    document.add_text(label=DocItemLabel.TITLE, text="   Sports science Evidence   ")
    extract_title_result = chunking.extract_title(document)
    assert extract_title_result == "Sports science Evidence"


@pytest.mark.parametrize("blank_title", [None, "", " \n\t "])
def test_no_usable_title_but_has_header(blank_title: str | None) -> None:
    document = DoclingDocument(name="example")
    if blank_title is not None:
        document.add_text(label=DocItemLabel.TITLE, text=blank_title)
    document.add_heading(text=" Sports science Evidence as a heading    ", level=1)
    extract_title_result = chunking.extract_title(document)
    assert extract_title_result == "Sports science Evidence as a heading"


@pytest.mark.parametrize("blank_title", ["", " \n\t "])
def test_blank_title_gets_ignored(blank_title: str) -> None:
    document = DoclingDocument(name="example")
    document.add_text(label=DocItemLabel.TITLE, text=blank_title)
    document.add_text(label=DocItemLabel.TITLE, text="   Sports science Evidence   ")
    extract_title_result = chunking.extract_title(document)
    assert extract_title_result == "Sports science Evidence"


def test_deeper_headings_or_text_gets_ignored_as_title() -> None:
    document = DoclingDocument(name="example")
    document.add_heading(text="Introduction", level=2)
    document.add_text(label=DocItemLabel.TEXT, text="   Sports science Evidence   ")
    extract_title_result = chunking.extract_title(document)
    assert extract_title_result is None


def test_empty_document_has_no_title() -> None:
    document = DoclingDocument(name="example")
    extract_title_result = chunking.extract_title(document)
    assert extract_title_result is None


def test_extract_title_returns_first_nonblank_title() -> None:
    document = DoclingDocument(name="example")
    document.add_text(label=DocItemLabel.TITLE, text="First title")
    document.add_text(label=DocItemLabel.TITLE, text="Second title")

    assert chunking.extract_title(document) == "First title"


def test_extract_title_skips_blank_level_one_heading() -> None:
    document = DoclingDocument(name="example")
    document.add_heading(text=" \n\t ", level=1)
    document.add_heading(text=" Results ", level=1)

    assert chunking.extract_title(document) == "Results"


def test_successful_chunking_and_coordination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "paper.json"
    document = DoclingDocument(name="Example")
    document.add_text(label=DocItemLabel.TITLE, text="Sport science evidence")

    fake_loader = Mock(return_value=document)
    fake_tokenizer = Mock()
    fake_chunker = Mock()

    fake_build_tokenizer = Mock(return_value=fake_tokenizer)
    fake_build_chunker = Mock(return_value=fake_chunker)

    first_chunk = Mock()
    second_chunk = Mock()
    fake_chunker.chunk.return_value = (first_chunk, second_chunk)
    fake_chunker.contextualize.side_effect = [
        "First contextualized chunk",
        "Second contextualized chunk",
    ]

    monkeypatch.setattr(chunking.DoclingDocument, "load_from_json", fake_loader)

    monkeypatch.setattr(chunking, "build_tokenizer", fake_build_tokenizer)

    monkeypatch.setattr(chunking, "build_chunker", fake_build_chunker)

    result = chunking.chunk_document(json_path, 256)

    fake_loader.assert_called_once_with(json_path)
    fake_build_tokenizer.assert_called_once_with(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        max_tokens=256,
    )

    fake_build_chunker.assert_called_once_with(fake_tokenizer)
    fake_chunker.chunk.assert_called_once_with(document)
    assert fake_chunker.contextualize.call_args_list == [
        call(chunk=first_chunk),
        call(chunk=second_chunk),
    ]

    assert result == (
        ["First contextualized chunk", "Second contextualized chunk"],
        "Sport science evidence",
    )


@pytest.mark.parametrize("title", ["Sports science evidence", None])
def test_chunk_document_returns_no_chunks_when_chunker_produces_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    title: str | None,
) -> None:
    document = DoclingDocument(name="example")
    if title is not None:
        document.add_text(label=DocItemLabel.TITLE, text=title)

    fake_tokenizer = Mock()
    fake_build_tokenizer = Mock(return_value=fake_tokenizer)
    fake_chunker = Mock()
    fake_chunker.chunk.return_value = iter(())
    monkeypatch.setattr(
        chunking.DoclingDocument, "load_from_json", Mock(return_value=document)
    )
    monkeypatch.setattr(chunking, "build_tokenizer", fake_build_tokenizer)
    monkeypatch.setattr(chunking, "build_chunker", Mock(return_value=fake_chunker))

    result = chunking.chunk_document(tmp_path / "paper.json")

    assert result == ([], title)
    fake_chunker.chunk.assert_called_once_with(document)
    fake_chunker.contextualize.assert_not_called()
    fake_build_tokenizer.assert_called_once_with(
        model_name="Qwen/Qwen3-Embedding-0.6B", max_tokens=1024
    )


def test_chunk_document_propagates_loading_errors_before_building_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "missing.json"
    error = FileNotFoundError("Document is missing")
    fake_loader = Mock(side_effect=error)
    fake_build_tokenizer = Mock()
    fake_build_chunker = Mock()
    monkeypatch.setattr(chunking.DoclingDocument, "load_from_json", fake_loader)
    monkeypatch.setattr(chunking, "build_tokenizer", fake_build_tokenizer)
    monkeypatch.setattr(chunking, "build_chunker", fake_build_chunker)

    with pytest.raises(FileNotFoundError) as raised:
        chunking.chunk_document(json_path)

    assert raised.value is error
    fake_loader.assert_called_once_with(json_path)
    fake_build_tokenizer.assert_not_called()
    fake_build_chunker.assert_not_called()


def test_chunk_document_propagates_chunk_generation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = DoclingDocument(name="example")
    error = RuntimeError("Chunk generation failed")
    fake_chunker = Mock()
    fake_chunker.chunk.side_effect = error
    monkeypatch.setattr(
        chunking.DoclingDocument, "load_from_json", Mock(return_value=document)
    )
    monkeypatch.setattr(chunking, "build_tokenizer", Mock())
    monkeypatch.setattr(chunking, "build_chunker", Mock(return_value=fake_chunker))

    with pytest.raises(RuntimeError) as raised:
        chunking.chunk_document(tmp_path / "paper.json")

    assert raised.value is error
    fake_chunker.chunk.assert_called_once_with(document)
    fake_chunker.contextualize.assert_not_called()


def test_chunk_document_propagates_errors_while_iterating_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = RuntimeError("Chunk iteration failed")
    first_chunk = Mock()

    def failing_chunks() -> Iterator[Mock]:
        yield first_chunk
        raise error

    fake_chunker = Mock()
    fake_chunker.chunk.return_value = failing_chunks()
    fake_chunker.contextualize.return_value = "First contextualized chunk"
    monkeypatch.setattr(
        chunking.DoclingDocument,
        "load_from_json",
        Mock(return_value=DoclingDocument(name="example")),
    )
    monkeypatch.setattr(chunking, "build_tokenizer", Mock())
    monkeypatch.setattr(chunking, "build_chunker", Mock(return_value=fake_chunker))

    with pytest.raises(RuntimeError) as raised:
        chunking.chunk_document(tmp_path / "paper.json")

    assert raised.value is error
    fake_chunker.contextualize.assert_called_once_with(chunk=first_chunk)


def test_chunk_document_propagates_contextualization_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_chunk = Mock()
    second_chunk = Mock()
    error = RuntimeError("Contextualization failed")
    fake_chunker = Mock()
    fake_chunker.chunk.return_value = [first_chunk, second_chunk]
    fake_chunker.contextualize.side_effect = error
    monkeypatch.setattr(
        chunking.DoclingDocument,
        "load_from_json",
        Mock(return_value=DoclingDocument(name="example")),
    )
    monkeypatch.setattr(chunking, "build_tokenizer", Mock())
    monkeypatch.setattr(chunking, "build_chunker", Mock(return_value=fake_chunker))

    with pytest.raises(RuntimeError) as raised:
        chunking.chunk_document(tmp_path / "paper.json")

    assert raised.value is error
    fake_chunker.contextualize.assert_called_once_with(chunk=first_chunk)


def test_build_tokenizer_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tokenizer = Mock()
    fake_from_pretrained = Mock(return_value=fake_tokenizer)
    monkeypatch.setattr(
        chunking.HuggingFaceTokenizer, "from_pretrained", fake_from_pretrained
    )

    result = chunking.build_tokenizer()

    assert result is fake_tokenizer
    fake_from_pretrained.assert_called_once_with(
        model_name="Qwen/Qwen3-Embedding-0.6B", max_tokens=1024
    )


def test_build_tokenizer_forwards_custom_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tokenizer = Mock()
    fake_from_pretrained = Mock(return_value=fake_tokenizer)
    monkeypatch.setattr(
        chunking.HuggingFaceTokenizer, "from_pretrained", fake_from_pretrained
    )

    result = chunking.build_tokenizer(model_name="example/tokenizer", max_tokens=256)

    assert result is fake_tokenizer
    fake_from_pretrained.assert_called_once_with(
        model_name="example/tokenizer", max_tokens=256
    )


def test_build_chunker_configures_tokenizer_and_table_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tokenizer = Mock()
    fake_provider = Mock()
    fake_provider_factory = Mock(return_value=fake_provider)
    fake_chunker = Mock()
    fake_hybrid_chunker = Mock(return_value=fake_chunker)
    monkeypatch.setattr(chunking, "MarkdownTableProvider", fake_provider_factory)
    monkeypatch.setattr(chunking, "HybridChunker", fake_hybrid_chunker)

    result = chunking.build_chunker(fake_tokenizer)

    assert result is fake_chunker
    fake_provider_factory.assert_called_once_with()
    fake_hybrid_chunker.assert_called_once_with(
        tokenizer=fake_tokenizer,
        serializer_provider=fake_provider,
        repeat_table_header=True,
        merge_peers=True,
    )


def test_markdown_table_provider_builds_compact_table_serializer() -> None:
    document = DoclingDocument(name="example")

    serializer = chunking.MarkdownTableProvider().get_serializer(document)

    assert serializer.doc is document
    assert isinstance(serializer.table_serializer, chunking.MarkdownTableSerializer)
    assert serializer.params.compact_tables is True


def test_chunk_document_with_real_json_chunker_and_local_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build a small tokenizer locally: this test never downloads model files.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = chunking.HuggingFaceTokenizer(
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]"),
        max_tokens=128,
    )
    monkeypatch.setattr(chunking, "build_tokenizer", Mock(return_value=tokenizer))

    document = DoclingDocument(name="example")
    document.add_text(label=DocItemLabel.TITLE, text="Sports science evidence")
    document.add_heading(text="Results", level=1)
    document.add_text(label=DocItemLabel.TEXT, text="Training improves strength.")
    caption = document.add_text(
        label=DocItemLabel.CAPTION, text="Table 1 Training outcomes"
    )
    document.add_table(
        caption=caption,
        data=TableData(
            num_rows=2,
            num_cols=2,
            table_cells=[
                TableCell(
                    text=text,
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=column,
                    end_col_offset_idx=column + 1,
                    column_header=row == 0,
                )
                for row, values in enumerate(
                    [("Study", "Outcome"), ("Trial", "Strength")]
                )
                for column, text in enumerate(values)
            ],
        ),
    )
    json_path = tmp_path / "paper.json"
    document.save_as_json(json_path)

    texts, title = chunking.chunk_document(json_path, max_tokens=128)

    assert title == "Sports science evidence"
    assert texts
    assert all(isinstance(text, str) and text.strip() for text in texts)
    combined_text = "\n".join(texts)
    assert "Results" in combined_text
    assert "Training improves strength." in combined_text
    assert "Table 1 Training outcomes" in combined_text
    assert "Trial" in combined_text
    assert "Strength" in combined_text
    assert any(
        line.strip().startswith("|") and "Study" in line and "Outcome" in line
        for line in combined_text.splitlines()
    )
