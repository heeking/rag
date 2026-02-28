"""
RAG Demo – Chunking Strategy & Retrieval Mode Comparison

This script:
  1. Loads sample documents
  2. Splits them with different chunking strategies
  3. Builds ChromaDB collections
  4. Compares retrieval modes: vector / bm25 / hybrid (+ optional reranking)
  5. Prints side-by-side comparison of retrieval results & answers

Usage:
    python main.py                              # default: recursive + hybrid + rerank
    python main.py --strategy recursive         # single chunking strategy
    python main.py --retrieval hybrid           # single retrieval mode
    python main.py --no-rerank                  # disable reranking
    python main.py --compare-retrieval          # compare all 3 retrieval modes
    python main.py --interactive                # interactive Q&A mode
"""

from __future__ import annotations

import argparse
import sys
import io
import time
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import (
    DOCS_DIR, CHROMA_DIR, CHUNK_SIZES,
    RETRIEVAL_MODES, DEFAULT_RETRIEVAL_MODE,
    SEARCH_K, RERANK_TOP_K, RERANK_BACKEND,
)
from document_loader import load_documents_from_dir
from chunking import chunk_documents
from vector_store import get_embeddings, build_vector_store, reset_vector_store
from retriever import build_retriever
from reranker import get_reranker
from rag_chain import build_rag_chain

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console(force_terminal=True)

STRATEGIES = ["character", "recursive", "semantic", "paragraph"]

TEST_QUESTIONS = [
    "什么是RAG技术？它解决了什么问题？",
    "Python有哪些常用的数据结构？",
    "云计算的三种服务模型分别是什么？",
    "向量数据库在RAG系统中起什么作用？",
    "什么是文本分片？有哪些常见的分片策略？",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_chunk(strategy: str, documents, embeddings) -> List:
    kwargs = CHUNK_SIZES.get(strategy, {})
    return chunk_documents(
        documents,
        strategy=strategy,
        embeddings=embeddings if strategy == "semantic" else None,
        **kwargs,
    )


def build_store_for_strategy(strategy: str, chunks, embeddings):
    persist_dir = Path(CHROMA_DIR) / strategy
    return build_vector_store(
        chunks, embeddings,
        collection_name=f"rag_{strategy}",
        persist_directory=str(persist_dir),
    )


def print_chunk_stats(strategy: str, chunks):
    lengths = [len(c.page_content) for c in chunks]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    console.print(
        f"  [bold]{strategy:12s}[/bold] | "
        f"片段数: [cyan]{len(chunks):4d}[/cyan] | "
        f"平均长度: [green]{avg_len:6.0f}[/green] | "
        f"最短: [yellow]{min_len:4d}[/yellow] | "
        f"最长: [red]{max_len:4d}[/red]"
    )


def print_stats_table(all_chunks, strategies):
    import numpy as np
    table = Table(title="各策略分片统计", box=box.ROUNDED)
    table.add_column("策略", style="bold")
    table.add_column("片段数", justify="right")
    table.add_column("平均长度", justify="right")
    table.add_column("最短", justify="right")
    table.add_column("最长", justify="right")
    table.add_column("标准差", justify="right")

    for strat in strategies:
        chunks = all_chunks[strat]
        lengths = [len(c.page_content) for c in chunks]
        avg = sum(lengths) / len(lengths) if lengths else 0
        std = float(np.std(lengths)) if lengths else 0
        table.add_row(
            strat,
            str(len(chunks)),
            f"{avg:.0f}",
            str(min(lengths)) if lengths else "0",
            str(max(lengths)) if lengths else "0",
            f"{std:.0f}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    strategies: List[str],
    retrieval_modes: List[str],
    use_rerank: bool = True,
    interactive: bool = False,
):
    console.print(Panel(
        "[bold magenta]RAG Demo -- 分片策略 & 检索模式 对比实验[/bold magenta]",
        expand=False,
    ))

    # 1. Load documents
    console.print("\n[bold]1. 加载文档...[/bold]")
    documents = load_documents_from_dir(DOCS_DIR)
    console.print(f"   已加载 [cyan]{len(documents)}[/cyan] 个文档")

    # 2. Embedding model
    console.print("\n[bold]2. 初始化 Embedding 模型...[/bold]")
    embeddings = get_embeddings()
    console.print("   Embedding 模型加载完成 [green]OK[/green]")

    # 3. Reranker
    reranker = None
    if use_rerank:
        console.print("\n[bold]3. 初始化 Reranker...[/bold]")
        t0 = time.perf_counter()
        reranker = get_reranker(backend=RERANK_BACKEND, top_k=RERANK_TOP_K)
        elapsed = time.perf_counter() - t0
        console.print(f"   Cross-Encoder Reranker 加载完成 ({elapsed:.1f}s) [green]OK[/green]")
    else:
        console.print("\n[bold]3. Reranker: [yellow]已禁用[/yellow][/bold]")

    # 4. Chunk & build stores
    step = 4
    console.print(f"\n[bold]{step}. 分片 & 构建向量数据库...[/bold]")
    stores = {}
    all_chunks = {}

    for strat in strategies:
        console.print(f"\n   >> 策略: [bold yellow]{strat}[/bold yellow]")
        t0 = time.perf_counter()
        chunks = load_and_chunk(strat, documents, embeddings)
        all_chunks[strat] = chunks
        print_chunk_stats(strat, chunks)

        store = build_store_for_strategy(strat, chunks, embeddings)
        stores[strat] = store
        elapsed = time.perf_counter() - t0
        console.print(f"     向量数据库构建完成 ({elapsed:.1f}s)")

    # 5. Stats
    step += 1
    console.print(f"\n[bold]{step}. 分片统计汇总[/bold]")
    print_stats_table(all_chunks, strategies)

    # 6. Q&A
    step += 1
    if interactive:
        _interactive_mode(stores, all_chunks, strategies, retrieval_modes, reranker)
    else:
        _batch_comparison(stores, all_chunks, strategies, retrieval_modes, reranker, step)


# ---------------------------------------------------------------------------
# Batch comparison
# ---------------------------------------------------------------------------

def _batch_comparison(stores, all_chunks, strategies, retrieval_modes, reranker, step):
    console.print(f"\n[bold]{step}. 问答对比测试[/bold]")
    mode_labels = {
        "vector": "向量检索",
        "bm25": "BM25关键词",
        "hybrid": "混合检索(BM25+向量)",
    }

    for q_idx, question in enumerate(TEST_QUESTIONS, 1):
        console.print(f"\n{'='*80}")
        console.print(f"[bold cyan]问题 {q_idx}: {question}[/bold cyan]")
        console.print(f"{'='*80}")

        for strat in strategies:
            for mode in retrieval_modes:
                retriever = build_retriever(
                    stores[strat], all_chunks[strat], mode=mode, search_k=SEARCH_K,
                )
                chain, retrieve_fn = build_rag_chain(retriever, reranker=reranker)

                docs = retrieve_fn(question)

                rerank_tag = "+Rerank" if reranker else ""
                label = f"{strat} | {mode_labels.get(mode, mode)}{rerank_tag}"
                console.print(f"\n  [bold yellow]<<{label}>>[/bold yellow]")
                console.print(f"  检索到 {len(docs)} 个相关片段:")
                for i, doc in enumerate(docs):
                    snippet = doc.page_content[:120].replace("\n", " ")
                    console.print(f"    \\[{i+1}] (长度 {len(doc.page_content)}) {snippet}...")

                if callable(chain):
                    answer = chain(question)
                else:
                    answer = chain.invoke(question)

                answer_preview = answer[:300].replace("\n", " ")
                console.print(f"  [green]回答: {answer_preview}[/green]")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def _interactive_mode(stores, all_chunks, strategies, retrieval_modes, reranker):
    console.print("\n[bold]交互模式[/bold] -- 输入问题进行检索，输入 'quit' 退出")
    console.print(f"   分片策略: {strategies}")
    console.print(f"   检索模式: {retrieval_modes}")
    console.print(f"   Reranker: {'ON' if reranker else 'OFF'}\n")

    mode_labels = {
        "vector": "向量检索",
        "bm25": "BM25关键词",
        "hybrid": "混合检索(BM25+向量)",
    }

    while True:
        question = console.input("[bold cyan]请输入问题: [/bold cyan]").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        for strat in strategies:
            for mode in retrieval_modes:
                retriever = build_retriever(
                    stores[strat], all_chunks[strat], mode=mode, search_k=SEARCH_K,
                )
                chain, retrieve_fn = build_rag_chain(retriever, reranker=reranker)
                docs = retrieve_fn(question)

                rerank_tag = "+Rerank" if reranker else ""
                label = f"{strat} | {mode_labels.get(mode, mode)}{rerank_tag}"
                console.print(f"\n  [bold yellow]<<{label}>>[/bold yellow]")
                for i, doc in enumerate(docs):
                    snippet = doc.page_content[:150].replace("\n", " ")
                    console.print(f"    \\[{i+1}] (长度 {len(doc.page_content)}) {snippet}...")

                if callable(chain):
                    answer = chain(question)
                else:
                    answer = chain.invoke(question)
                console.print(f"  [green]回答: {answer}[/green]\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG Chunking & Retrieval Comparison Demo")
    parser.add_argument(
        "--strategy", choices=STRATEGIES, default=None,
        help="Run a single chunking strategy (default: recursive)",
    )
    parser.add_argument(
        "--retrieval", choices=RETRIEVAL_MODES, default=None,
        help="Run a single retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--compare-retrieval", action="store_true",
        help="Compare all 3 retrieval modes (vector, bm25, hybrid)",
    )
    parser.add_argument(
        "--compare-chunking", action="store_true",
        help="Compare all 4 chunking strategies",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Disable reranking",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter interactive Q&A mode",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing vector stores before building",
    )
    args = parser.parse_args()

    if args.reset:
        console.print("[red]Resetting vector stores...[/red]")
        reset_vector_store()

    if args.compare_chunking:
        strategies = STRATEGIES
    elif args.strategy:
        strategies = [args.strategy]
    else:
        strategies = ["recursive"]

    if args.compare_retrieval:
        retrieval_modes = RETRIEVAL_MODES
    elif args.retrieval:
        retrieval_modes = [args.retrieval]
    else:
        retrieval_modes = [DEFAULT_RETRIEVAL_MODE]

    run_pipeline(
        strategies=strategies,
        retrieval_modes=retrieval_modes,
        use_rerank=not args.no_rerank,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
