"""
Re-ranking module.

Provides two reranker backends:
  1. CrossEncoderReranker — local cross-encoder model (no API key needed)
  2. LLMReranker          — uses the configured LLM to score relevance

Cross-encoder models are specifically trained to judge query-document relevance
and typically outperform bi-encoder (embedding) similarity for ranking.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from config import RERANK_MODEL_NAME, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


class CrossEncoderReranker:
    """Rerank documents using a local cross-encoder model."""

    def __init__(self, model_name: str = RERANK_MODEL_NAME, top_k: int = 3):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return documents

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )
        return [doc for doc, _ in scored_docs[: self.top_k]]


class LLMReranker:
    """Rerank documents by asking the LLM to score relevance (0-10)."""

    def __init__(self, top_k: int = 3):
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0,
        )
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return documents

        scored_docs = []
        for doc in documents:
            prompt = (
                f"请评估以下文档与查询的相关性，只返回一个0到10的整数分数。\n"
                f"查询: {query}\n"
                f"文档: {doc.page_content[:500]}\n"
                f"相关性分数:"
            )
            try:
                resp = self.llm.invoke(prompt)
                score = float(resp.content.strip())
            except (ValueError, AttributeError):
                score = 0.0
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[: self.top_k]]


def get_reranker(backend: str = "cross-encoder", top_k: int = 3):
    """
    Factory function to create a reranker.

    Parameters
    ----------
    backend : str
        "cross-encoder" — local cross-encoder model (default, no API needed)
        "llm"           — use the configured LLM for reranking
        "none"          — no reranking, return None
    """
    if backend == "none":
        return None
    if backend == "cross-encoder":
        return CrossEncoderReranker(top_k=top_k)
    if backend == "llm":
        return LLMReranker(top_k=top_k)
    raise ValueError(f"Unknown reranker backend: '{backend}'. Choose from: cross-encoder, llm, none")
