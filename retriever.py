"""
Retrieval strategies: vector-only, BM25-only, and hybrid (BM25 + vector).

Hybrid retrieval fuses keyword-based BM25 scores with dense vector similarity
using Reciprocal Rank Fusion (RRF), combining the strengths of both approaches.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi


def _tokenize_chinese(text: str) -> List[str]:
    """Simple character-level + word-level tokenizer for Chinese text."""
    text = text.lower()
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)
    return tokens


class BM25Retriever(BaseRetriever):
    """BM25 keyword retriever over a list of LangChain Documents."""

    documents: List[Document]
    bm25: object = None  # BM25Okapi instance
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, documents: List[Document], k: int = 4, **kwargs):
        super().__init__(documents=documents, k=k, **kwargs)
        corpus = [_tokenize_chinese(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        tokens = _tokenize_chinese(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        return [self.documents[i] for i, _ in ranked[: self.k]]


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever using Reciprocal Rank Fusion (RRF).

    Merges results from a vector retriever and a BM25 retriever.
    RRF score = sum(1 / (rrf_k + rank_i)) across all retrievers.
    """

    vector_retriever: object  # Chroma retriever
    bm25_retriever: BM25Retriever
    k: int = 4
    rrf_k: int = 60  # RRF constant, standard default

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        vec_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(vec_docs):
            key = doc.page_content[:200]
            doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:200]
            doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys[: self.k]]


def build_retriever(
    vector_store: Chroma,
    chunks: List[Document],
    mode: str = "vector",
    search_k: int = 4,
):
    """
    Build a retriever based on the specified mode.

    Parameters
    ----------
    mode : str
        "vector"  — pure vector similarity search
        "bm25"    — pure BM25 keyword search
        "hybrid"  — RRF fusion of vector + BM25
    """
    if mode == "vector":
        return vector_store.as_retriever(search_kwargs={"k": search_k})

    bm25 = BM25Retriever(documents=chunks, k=search_k)

    if mode == "bm25":
        return bm25

    if mode == "hybrid":
        vec_retriever = vector_store.as_retriever(search_kwargs={"k": search_k})
        return HybridRetriever(
            vector_retriever=vec_retriever,
            bm25_retriever=bm25,
            k=search_k,
        )

    raise ValueError(f"Unknown retrieval mode: '{mode}'. Choose from: vector, bm25, hybrid")
