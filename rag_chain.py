"""
RAG chain construction.

Provides two modes:
  - OpenAI-based LLM (requires API key)
  - Fake/stub LLM for local-only demo without any API
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


RAG_PROMPT_TEMPLATE = """\
你是一个专业的问答助手。请根据以下检索到的上下文内容回答用户的问题。
如果上下文中没有相关信息，请如实说明你无法根据已有资料回答。

## 检索到的上下文
{context}

## 用户问题
{question}

## 回答
"""


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[来源: {d.metadata.get('source', '未知')}, 片段 #{d.metadata.get('chunk_index', '?')}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain(vector_store: Chroma, search_k: int = 4):
    """
    Build a Retrieval-Augmented Generation chain.

    If an OpenAI-compatible API key is configured, uses ChatOpenAI.
    Otherwise falls back to a simple context-only mode that returns
    the retrieved chunks directly (no LLM generation).
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": search_k})
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    if LLM_API_KEY:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=LLM_MODEL_NAME,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=0,
        )
        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        def _no_llm_fallback(question: str) -> str:
            docs = retriever.invoke(question)
            context = _format_docs(docs)
            return (
                f"[无 LLM 模式 — 仅展示检索结果]\n\n"
                f"问题: {question}\n\n"
                f"检索到的上下文:\n{context}"
            )

        chain = _no_llm_fallback

    return chain, retriever
