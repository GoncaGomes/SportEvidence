import os
from pathlib import Path

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.text import SectionHeaderItem
from docling_core.types.doc.labels import DocItemLabel

os.environ.get("HF_TOKEN", None)


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


def extract_title(document: DoclingDocument) -> str | None:
    for item in document.texts:
        if item.label == DocItemLabel.TITLE and item.text.strip():
            return item.text.strip()

    for item in document.texts:
        if (
            isinstance(item, SectionHeaderItem)
            and item.level == 1
            and item.text.strip()
        ):
            return item.text.strip()

    return None


def build_chunker(
    tokenizer: HuggingFaceTokenizer,
) -> HybridChunker:
    return HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=MarkdownTableProvider(),
        repeat_table_header=True,
        merge_peers=True,
    )


def chunk_document(json_path: Path, max_tokens: int = 1024) -> tuple[list, str | None]:
    document = DoclingDocument.load_from_json(json_path)
    tokenizer = build_tokenizer(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        max_tokens=max_tokens,
    )
    chunker = build_chunker(tokenizer)
    chunks_text = [
        chunker.contextualize(chunk=chunks) for chunks in chunker.chunk(document)
    ]
    return chunks_text, extract_title(document)
