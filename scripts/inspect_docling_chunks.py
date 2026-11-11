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

JSON_PATH = Path("tests/fixtures/corpus_json/SPORT-002_prepared.json")


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
        print(
            f"Chunk {index}:\nSize in tokens: {token_number}\n", f"Text{embeddings}\n\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
