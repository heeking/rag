"""
Vector store management with ChromaDB.

Handles embedding creation, collection management, and similarity search.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, EMBEDDING_MODEL_NAME


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Create a HuggingFace sentence-transformer embedding function."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str = "rag_demo",
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Embed chunks and persist them into a Chroma collection."""
    persist_dir = str(persist_directory or CHROMA_DIR)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return store


def load_vector_store(
    embeddings: HuggingFaceEmbeddings,
    collection_name: str = "rag_demo",
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Load an existing Chroma collection from disk."""
    persist_dir = str(persist_directory or CHROMA_DIR)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def reset_vector_store(persist_directory: str | Path | None = None) -> None:
    """Delete the persisted vector store directory."""
    persist_dir = Path(persist_directory or CHROMA_DIR)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
