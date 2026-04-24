"""
LLM Inference Benchmark Orchestrator (SAFE VERSION)
Fixes:
- no infinite hangs
- per-request timeout
- safe concurrency handling
- AirLLM isolation
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/results"))
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
NUM_TOKENS = int(os.getenv("NUM_TOKENS", "256"))
WARMUP = int(os.getenv("WARMUP_REQUESTS", "2"))
CONCURRENCY_LEVELS = [int(x) for x in os.getenv("CONCURRENCY", "1,4,8").split(",")]

ENGINES_FILTER = (
    set(os.getenv("ENGINES_FILTER", "").split(","))
    if os.getenv("ENGINES_FILTER")
    else None
)

REQUEST_TIMEOUT = 120
WARMUP_TIMEOUT = 60

# ─────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    engine: str
    concurrency: int
    success: bool
    ttft_ms: float
    total_latency_ms: float
    output_tokens: int
    error: Optional[str] = None


@dataclass
class BenchmarkStats:
    engine: str
    concurrency: int
    n_requests: int
    n_success: int
    throughput_tok_s: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    latency_p95_ms: float


# ─────────────────────────────────────────────────────────────
# ENGINE ADAPTERS (INTERFACE SIMPLIFIÉE)
# ─────────────────────────────────────────────────────────────

class EngineAdapter:
    name: str
    base_url: str

    async def generate(self, client, prompt: str, max_tokens: int) -> RequestResult:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# SAFE BENCH ENGINE RUNNER
# ─────────────────────────────────────────────────────────────

async def safe_call(adapter, client, prompt, max_tokens):
    try:
        return await asyncio.wait_for(
            adapter.generate(client, prompt, max_tokens),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return RequestResult(
            engine=adapter.name,
            concurrency=0,
            success=False,
            ttft_ms=0,
            total_latency_ms=REQUEST_TIMEOUT * 1000,
            output_tokens=0,
            error="timeout",
        )
    except Exception as e:
        return RequestResult(
            engine=adapter.name,
            concurrency=0,
            success=False,
            ttft_ms=0,
            total_latency_ms=0,
            output_tokens=0,
            error=str(e),
        )


# ─────────────────────────────────────────────────────────────
# CORE BENCH
# ─────────────────────────────────────────────────────────────

async def bench_engine(adapter, prompts, concurrency, max_tokens, warmup):
    timeout = httpx.Timeout(300.0)

    async with httpx.AsyncClient(timeout=timeout) as client:

        # ── WARMUP SAFE ─────────────────────────────
        for i in range(warmup):
            try:
                await asyncio.wait_for(
                    adapter.generate(client, prompts[i % len(prompts)], max_tokens),
                    timeout=WARMUP_TIMEOUT,
                )
            except Exception:
                pass

        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def run_one(prompt):
            async with semaphore:
                r = await safe_call(adapter, client, prompt, max_tokens)
                r.concurrency = concurrency
                results.append(r)

        # ── LAUNCH REQUESTS ─────────────────────────
        tasks = [
            run_one(prompts[i % len(prompts)])
            for i in range(len(prompts))
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    return compute_stats(results, adapter.name, concurrency)


# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────

def compute_stats(results, engine, concurrency):
    ok = [r for r in results if r.success]

    def pct(arr, p):
        return float(np.percentile(arr, p)) if arr else 0.0

    ttfts = [r.ttft_ms for r in ok]
    lats = [r.total_latency_ms for r in ok]
    tokens = sum(r.output_tokens for r in ok)

    total_time = sum(lats) / 1000 if lats else 0
    throughput = tokens / total_time if total_time > 0 else 0

    return BenchmarkStats(
        engine=engine,
        concurrency=concurrency,
        n_requests=len(results),
        n_success=len(ok),
        throughput_tok_s=round(throughput, 2),
        ttft_p50_ms=pct(ttfts, 50),
        ttft_p95_ms=pct(ttfts, 95),
        latency_p95_ms=pct(lats, 95),
    )


# ─────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────

async def main():
    console.print("[bold cyan]Benchmark SAFE MODE[/bold cyan]")
    console.print(f"Model: {MODEL_NAME}")
    console.print(f"Concurrency: {CONCURRENCY_LEVELS}")
    console.print()

    # NOTE: adapters injectés depuis ton code existant
    from runners import adapters  # suppose ton fichier actuel

    if ENGINES_FILTER:
        adapters = [a for a in adapters if a.name in ENGINES_FILTER]

    all_stats = []
    total = len(adapters) * len(CONCURRENCY_LEVELS)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Running benchmark", total=total)

        for adapter in adapters:
            for c in CONCURRENCY_LEVELS:

                # 🔥 CRITICAL FIX: AirLLM forced single concurrency
                if adapter.name == "airllm":
                    c = 1

                progress.update(task, description=f"{adapter.name} @ c={c}")

                stats = await bench_engine(
                    adapter,
                    load_prompts(),
                    c,
                    NUM_TOKENS,
                    WARMUP,
                )

                all_stats.append(stats)
                progress.advance(task)

    print_table(all_stats)
    save_results(all_stats)


# ─────────────────────────────────────────────────────────────
# HELPERS (simplifiés)
# ─────────────────────────────────────────────────────────────

def load_prompts():
    return [
        "Explain transformers in simple terms.",
        "Write a Python quicksort.",
        "What is Kubernetes?",
        "Summarize machine learning.",
    ]


def print_table(stats):
    table = Table(title="Benchmark Results")

    table.add_column("Engine")
    table.add_column("c")
    table.add_column("Success")
    table.add_column("Tok/s")
    table.add_column("TTFT p95")
    table.add_column("Lat p95")

    for s in stats:
        table.add_row(
            s.engine,
            str(s.concurrency),
            f"{s.n_success}/{s.n_requests}",
            str(s.throughput_tok_s),
            str(round(s.ttft_p95_ms, 1)),
            str(round(s.latency_p95_ms, 1)),
        )

    console.print(table)


def save_results(stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    path = OUTPUT_DIR / f"results_{ts}.json"

    with open(path, "w") as f:
        json.dump([asdict(s) for s in stats], f, indent=2)

    console.print(f"[green]Saved to {path}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
    