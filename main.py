"""
RAG Demo – Chunking Strategy Comparison

This script:
  1. Loads sample documents
  2. Splits them with four different chunking strategies
  3. Builds a separate ChromaDB collection for each strategy
  4. Runs the same set of test questions against each collection
  5. Prints a side-by-side comparison of retrieval results & answers

Usage:
    python main.py                  # run full comparison
    python main.py --strategy semantic  # run a single strategy
    python main.py --interactive    # interactive Q&A mode
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import DOCS_DIR, CHROMA_DIR, CHUNK_SIZES
from document_loader import load_documents_from_dir
from chunking import chunk_documents, STRATEGY_MAP
from vector_store import get_embeddings, build_vector_store, reset_vector_store
from rag_chain import build_rag_chain

import sys
import io

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


def load_and_chunk(
    strategy: str,
    documents,
    embeddings,
) -> List:
    """Chunk documents with the given strategy and return chunks."""
    kwargs = CHUNK_SIZES.get(strategy, {})
    chunks = chunk_documents(
        documents,
        strategy=strategy,
        embeddings=embeddings if strategy == "semantic" else None,
        **kwargs,
    )
    return chunks


def build_store_for_strategy(strategy: str, chunks, embeddings):
    """Build a ChromaDB collection for one strategy."""
    persist_dir = Path(CHROMA_DIR) / strategy
    store = build_vector_store(
        chunks,
        embeddings,
        collection_name=f"rag_{strategy}",
        persist_directory=str(persist_dir),
    )
    return store


def print_chunk_stats(strategy: str, chunks):
    """Print statistics about the chunks."""
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


def run_comparison(strategies: List[str], interactive: bool = False):
    """Main comparison pipeline."""

    console.print(Panel("[bold magenta]RAG Demo — 分片策略对比实验[/bold magenta]", expand=False))

    # --- Load documents ---
    console.print("\n[bold]1. 加载文档...[/bold]")
    documents = load_documents_from_dir(DOCS_DIR)
    console.print(f"   已加载 [cyan]{len(documents)}[/cyan] 个文档")

    # --- Embedding model ---
    console.print("\n[bold]2. 初始化 Embedding 模型...[/bold]")
    embeddings = get_embeddings()
    console.print("   Embedding 模型加载完成 [green]OK[/green]")

    # --- Chunk & build stores ---
    console.print("\n[bold]3. 分片 & 构建向量数据库...[/bold]")
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

    # --- Chunk statistics summary ---
    console.print("\n[bold]4. 分片统计汇总[/bold]")
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
        import numpy as np
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

    # --- Q&A comparison ---
    if interactive:
        _interactive_mode(stores, strategies)
    else:
        _batch_comparison(stores, strategies)


def _batch_comparison(stores, strategies):
    """Run test questions against all strategies and compare."""
    console.print("\n[bold]5. 问答对比测试[/bold]")

    for q_idx, question in enumerate(TEST_QUESTIONS, 1):
        console.print(f"\n{'='*80}")
        console.print(f"[bold cyan]问题 {q_idx}: {question}[/bold cyan]")
        console.print(f"{'='*80}")

        for strat in strategies:
            chain, retriever = build_rag_chain(stores[strat], search_k=3)
            docs = retriever.invoke(question)

            console.print(f"\n  [bold yellow]【{strat}】策略[/bold yellow]")
            console.print(f"  检索到 {len(docs)} 个相关片段:")
            for i, doc in enumerate(docs):
                snippet = doc.page_content[:120].replace("\n", " ")
                console.print(f"    [{i+1}] (长度 {len(doc.page_content)}) {snippet}...")

            if callable(chain):
                answer = chain(question)
            else:
                answer = chain.invoke(question)

            answer_preview = answer[:300].replace("\n", " ")
            console.print(f"  [green]回答: {answer_preview}[/green]")


def _interactive_mode(stores, strategies):
    """Interactive Q&A loop."""
    console.print("\n[bold]交互模式[/bold] — 输入问题进行检索，输入 'quit' 退出\n")

    while True:
        question = console.input("[bold cyan]请输入问题: [/bold cyan]").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        for strat in strategies:
            chain, retriever = build_rag_chain(stores[strat], search_k=3)
            docs = retriever.invoke(question)

            console.print(f"\n  [bold yellow]【{strat}】策略[/bold yellow]")
            for i, doc in enumerate(docs):
                snippet = doc.page_content[:150].replace("\n", " ")
                console.print(f"    [{i+1}] (长度 {len(doc.page_content)}) {snippet}...")

            if callable(chain):
                answer = chain(question)
            else:
                answer = chain.invoke(question)
            console.print(f"  [green]回答: {answer}[/green]\n")


def main():
    parser = argparse.ArgumentParser(description="RAG Chunking Strategy Comparison Demo")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=None,
        help="Run a single strategy instead of all four",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive Q&A mode after building indexes",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing vector stores before building",
    )
    args = parser.parse_args()

    if args.reset:
        console.print("[red]Resetting vector stores...[/red]")
        reset_vector_store()

    strategies = [args.strategy] if args.strategy else STRATEGIES
    run_comparison(strategies, interactive=args.interactive)


if __name__ == "__main__":
    main()
