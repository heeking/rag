"""
Text chunking strategies for RAG.

Implements four strategies to study how splitting granularity affects retrieval quality:
  1. Character-based  – fixed window by character count
  2. Recursive character – recursively splits on natural boundaries (\n\n, \n, sentence, word)
  3. Semantic          – groups consecutive sentences whose embeddings are similar
  4. Paragraph-based   – splits on blank-line paragraph boundaries
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


# ---------------------------------------------------------------------------
# 1. Character-based splitter
# ---------------------------------------------------------------------------

def split_by_character(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Naive fixed-window split by character count.

    Pros: simple, deterministic.
    Cons: may cut mid-sentence, losing semantic coherence.
    """
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_strategy"] = "character"
        c.metadata["chunk_index"] = i
    return chunks


# ---------------------------------------------------------------------------
# 2. Recursive character splitter
# ---------------------------------------------------------------------------

def split_by_recursive(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Recursively split on paragraph > newline > sentence > word boundaries.

    This is LangChain's recommended default splitter – it tries to keep
    semantically related text together while respecting the size limit.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_strategy"] = "recursive"
        c.metadata["chunk_index"] = i
    return chunks


# ---------------------------------------------------------------------------
# 3. Semantic splitter (embedding-based)
# ---------------------------------------------------------------------------

def split_by_semantic(
    documents: List[Document],
    embeddings,
    breakpoint_threshold_type: str = "percentile",
) -> List[Document]:
    """
    Group consecutive sentences whose embedding similarity exceeds a threshold.

    Requires an embedding model. Sentences that are semantically close stay
    in the same chunk, producing variable-length but coherent chunks.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_strategy"] = "semantic"
        c.metadata["chunk_index"] = i
    return chunks


# ---------------------------------------------------------------------------
# 4. Paragraph-based splitter
# ---------------------------------------------------------------------------

_PARA_RE = re.compile(r"\n\s*\n")


def split_by_paragraph(documents: List[Document]) -> List[Document]:
    """
    Split on blank-line boundaries (i.e. natural paragraphs).

    Preserves the author's original paragraph structure. Chunk sizes are
    uneven – some paragraphs may be very short or very long.
    """
    chunks: List[Document] = []
    idx = 0
    for doc in documents:
        paragraphs = _PARA_RE.split(doc.page_content)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            chunks.append(
                Document(
                    page_content=para,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": "paragraph",
                        "chunk_index": idx,
                    },
                )
            )
            idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "character": split_by_character,
    "recursive": split_by_recursive,
    "semantic": split_by_semantic,
    "paragraph": split_by_paragraph,
}


def chunk_documents(
    documents: List[Document],
    strategy: str = "recursive",
    embeddings=None,
    **kwargs,
) -> List[Document]:
    """
    Dispatch to the requested chunking strategy.

    Parameters
    ----------
    strategy : str
        One of "character", "recursive", "semantic", "paragraph".
    embeddings :
        Required only for the "semantic" strategy.
    **kwargs :
        Forwarded to the underlying splitter.
    """
    if strategy not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGY_MAP.keys())}")

    fn = STRATEGY_MAP[strategy]
    if strategy == "semantic":
        if embeddings is None:
            raise ValueError("semantic strategy requires an embeddings model")
        return fn(documents, embeddings=embeddings, **kwargs)
    if strategy == "paragraph":
        return fn(documents)
    return fn(documents, **kwargs)
