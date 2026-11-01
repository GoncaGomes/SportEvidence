import os
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc import DoclingDocument
from dotenv import load_dotenv

JSON_PATH = Path(
    "tests/fixtures/corpus_json/SPORT-002_prepared_associated.json"
)


class MarkdownTableProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


def build_tokenizer(
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    max_tokens: int = 1024,
) -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer.from_pretrained(
        model_name=model_name,
        max_tokens=max_tokens,
    )


def build_chunker(
    tokenizer: HuggingFaceTokenizer,
) -> HybridChunker:
    return HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=MarkdownTableProvider(),
        repeat_table_header=True,
        merge_peers=True,
    )


def main() -> None:
    load_dotenv()
    os.environ.get("HF_TOKEN", None)
    document = DoclingDocument.load_from_json(JSON_PATH)

    print(f"Document type: {type(document).__name__}")
    print(f"Prepared pages: {len(document.pages)}")
    print(f"Text items: {len(document.texts)}")
    print(f"Table items: {len(document.tables)}")

    tokenizer = build_tokenizer(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        max_tokens=1024,
    )

    chunker = build_chunker(tokenizer)

    chunks = list(chunker.chunk(dl_doc=document))

    print(f"\n\n{chunks[18].text}\n\n")
    print(f"\n\n{chunker.contextualize(chunk=chunks[18])}\n\n")

    print(f"\nTotal chunks: {len(chunks)}")
    {item.label for item in document.texts}
    print("==" * 50)
    print(chunks[23].model_dump_json(indent=2))
    print("==" * 50)
    embedding_chunck23 = chunker.contextualize(chunk=chunks[23])
    print(embedding_chunck23)


    for index, chunk in enumerate(chunks[:10], start=1):
        embeddings = chunker.contextualize(chunk=chunk)
        token_number = tokenizer.count_tokens(embeddings)
        print(f"Chunk {index}:Size in tokens: {token_number}Text{embeddings}\n\n")

    """
    for index, chunk in enumerate(chunks, start=1):
        items = chunk.meta.doc_items

        table_references.update(
            item.self_ref
            for item in items
            if item.label == DocItemLabel.TABLE
        )

        has_text = any(
            item.label in text_labels
            for item in items
        )
        has_table = any(
            item.label == DocItemLabel.TABLE
            for item in items
        )

        if not (has_text and has_table):
            continue

        mixed_chunk_count += 1

        references = [item.self_ref for item in items]
        embedding_text = chunker.contextualize(chunk=chunk)
        token_count = tokenizer.count_tokens(embedding_text)

        print(
            f"Chunk {index}: "
            f"tokens={token_count}, "
            f"references={references}"
        )

        if mixed_chunk_count <= 5:
            print(embedding_text)
            print()

    print(f"\nMixed chunks: {mixed_chunk_count}")
    print(f"Distinct table items represented: {len(table_references)}")

    expected_references = {table.self_ref for table in document.tables}
    missing_references = expected_references - table_references

    print(f"Table items without chunks: {sorted(missing_references)}")
    """


if __name__ == "__main__":
    raise SystemExit(main())
