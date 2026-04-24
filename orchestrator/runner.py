"""
LLM Inference Benchmark Orchestrator
Measures: TTFT, throughput (tok/s), latency (p50/p95/p99), GPU memory
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────

ENGINES: dict[str, "EngineAdapter"] = {}

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/results"))
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
NUM_TOKENS = int(os.getenv("NUM_TOKENS", "256"))
WARMUP = int(os.getenv("WARMUP_REQUESTS", "3"))
CONCURRENCY_LEVELS = [int(x) for x in os.getenv("CONCURRENCY", "1,4,8").split(",")]

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    engine: str
    concurrency: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float            # time-to-first-token
    total_latency_ms: float
    success: bool
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
    ttft_p99_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    total_output_tokens: int

# ── Engine adapters ───────────────────────────────────────────────────────────

class EngineAdapter:
    name: str
    base_url: str

    async def generate(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        max_tokens: int,
    ) -> RequestResult:
        raise NotImplementedError


class AirLLMAdapter(EngineAdapter):
    name = "airllm"

    def __init__(self):
        self.base_url = os.getenv("AIRLLM_URL", "http://airllm:8080")

    async def generate(self, client, prompt, max_tokens):
        payload = {"prompt": prompt, "max_new_tokens": max_tokens, "stream": True}
        t0 = time.perf_counter()
        ttft_ms = None
        total_tokens = 0
        try:
            async with client.stream("POST", f"{self.base_url}/generate", json=payload, timeout=120) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    data = json.loads(chunk)
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    total_tokens += data.get("token_count", 1)
            t1 = time.perf_counter()
            return RequestResult(
                engine=self.name, concurrency=0,
                prompt_tokens=len(prompt.split()),
                output_tokens=total_tokens,
                ttft_ms=ttft_ms or 0,
                total_latency_ms=(t1 - t0) * 1000,
                success=True,
            )
        except Exception as e:
            return RequestResult(
                engine=self.name, concurrency=0,
                prompt_tokens=0, output_tokens=0,
                ttft_ms=0, total_latency_ms=0,
                success=False, error=str(e),
            )


class LlamaCppAdapter(EngineAdapter):
    name = "llamacpp"

    def __init__(self):
        self.base_url = os.getenv("LLAMACPP_URL", "http://llamacpp:8080")

    async def generate(self, client, prompt, max_tokens):
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "stream": True,
            "temperature": 0.0,
            "cache_prompt": False,
        }
        t0 = time.perf_counter()
        ttft_ms = None
        total_tokens = 0
        try:
            async with client.stream("POST", f"{self.base_url}/completion", json=payload, timeout=120) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip())
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    total_tokens += 1
                    if data.get("stop"):
                        break
            t1 = time.perf_counter()
            return RequestResult(
                engine=self.name, concurrency=0,
                prompt_tokens=len(prompt.split()),
                output_tokens=total_tokens,
                ttft_ms=ttft_ms or 0,
                total_latency_ms=(t1 - t0) * 1000,
                success=True,
            )
        except Exception as e:
            return RequestResult(
                engine=self.name, concurrency=0,
                prompt_tokens=0, output_tokens=0,
                ttft_ms=0, total_latency_ms=0,
                success=False, error=str(e),
            )


class OllamaAdapter(EngineAdapter):
    name = "ollama"

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.model = "qwen2.5:7b" 

    async def generate(self, client, prompt, max_tokens):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
        t0 = time.perf_counter()
        ttft_ms = None
        total_tokens = 0
        try:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload, timeout=120) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    total_tokens += 1
                    if data.get("done"):
                        break
            t1 = time.perf_counter()
            return RequestResult(
                engine=self.name, concurrency=0,
                prompt_tokens=len(prompt.split()),
                output_tokens=total_tokens,
                ttft_ms=ttft_ms or 0,
                total_latency_ms=(t1 - t0) * 1000,
                success=True,
            )
        except Exception as e:
            return RequestResult(
                engine=self.name, concurrency=0,
                prompt_tokens=0, output_tokens=0,
                ttft_ms=0, total_latency_ms=0,
                success=False, error=str(e),
            )


class VLLMAdapter(EngineAdapter):
    name = "vllm"

    def __init__(self):
        self.base_url = os.getenv("VLLM_URL", "http://vllm:8000")
        self.model = "qwen2.5:7b"  # au lieu de MODEL_NAME


    async def generate(self, client, prompt, max_tokens):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        t0 = time.perf_counter()
        ttft_ms = None
        total_tokens = 0
        try:
            async with client.stream("POST", f"{self.base_url}/v1/completions", json=payload, timeout=120) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    data = json.loads(chunk)
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    for choice in data.get("choices", []):
                        total_tokens += len(choice.get("text", "").split())
            t1 = time.perf_counter()
            return RequestResult(
                engine=self.name, concurrency=0,
                prompt_tokens=len(prompt.split()),
                output_tokens=total_tokens,
                ttft_ms=ttft_ms or 0,
                total_latency_ms=(t1 - t0) * 1000,
                success=True,
            )
        except Exception as e:
            return RequestResult(
                engine=self.name, concurrency=0,
                prompt_tokens=0, output_tokens=0,
                ttft_ms=0, total_latency_ms=0,
                success=False, error=str(e),
            )


# ── Benchmark logic ───────────────────────────────────────────────────────────

def _percentile(arr: list[float], p: int) -> float:
    if not arr:
        return 0.0
    return float(np.percentile(arr, p))


def compute_stats(results: list[RequestResult], concurrency: int) -> BenchmarkStats:
    ok = [r for r in results if r.success]
    ttfts = [r.ttft_ms for r in ok]
    lats = [r.total_latency_ms for r in ok]
    total_tokens = sum(r.output_tokens for r in ok)
    total_time_s = sum(r.total_latency_ms for r in ok) / 1000.0
    throughput = total_tokens / total_time_s if total_time_s > 0 else 0

    return BenchmarkStats(
        engine=results[0].engine if results else "unknown",
        concurrency=concurrency,
        n_requests=len(results),
        n_success=len(ok),
        throughput_tok_s=round(throughput, 2),
        ttft_p50_ms=round(_percentile(ttfts, 50), 1),
        ttft_p95_ms=round(_percentile(ttfts, 95), 1),
        ttft_p99_ms=round(_percentile(ttfts, 99), 1),
        latency_p50_ms=round(_percentile(lats, 50), 1),
        latency_p95_ms=round(_percentile(lats, 95), 1),
        latency_p99_ms=round(_percentile(lats, 99), 1),
        total_output_tokens=total_tokens,
    )


async def bench_engine(
    adapter: EngineAdapter,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    warmup: int,
) -> BenchmarkStats:
    """Run benchmark for a single engine at a given concurrency level."""

    async with httpx.AsyncClient() as client:
        # warm-up
        for i in range(warmup):
            await adapter.generate(client, prompts[i % len(prompts)], max_tokens)

        # actual benchmark
        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []

        async def one_request(prompt: str) -> None:
            async with semaphore:
                r = await adapter.generate(client, prompt, max_tokens)
                r.concurrency = concurrency
                results.append(r)

        tasks = [one_request(prompts[i % len(prompts)]) for i in range(len(prompts))]
        await asyncio.gather(*tasks)

    return compute_stats(results, concurrency)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(all_stats: list[BenchmarkStats]) -> None:
    table = Table(title="[bold]Inference Benchmark Results[/bold]", show_lines=True)
    table.add_column("Engine", style="cyan")
    table.add_column("Concurrency", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Throughput\n(tok/s)", justify="right", style="green")
    table.add_column("TTFT p50\n(ms)", justify="right")
    table.add_column("TTFT p95\n(ms)", justify="right")
    table.add_column("Lat p50\n(ms)", justify="right")
    table.add_column("Lat p95\n(ms)", justify="right")
    table.add_column("Lat p99\n(ms)", justify="right")

    for s in all_stats:
        table.add_row(
            s.engine,
            str(s.concurrency),
            f"{s.n_success}/{s.n_requests}",
            str(s.throughput_tok_s),
            str(s.ttft_p50_ms),
            str(s.ttft_p95_ms),
            str(s.latency_p50_ms),
            str(s.latency_p95_ms),
            str(s.latency_p99_ms),
        )
    console.print(table)


def save_results(all_stats: list[BenchmarkStats]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    # JSON
    json_path = OUTPUT_DIR / f"results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump([asdict(s) for s in all_stats], f, indent=2)

    # CSV
    csv_path = OUTPUT_DIR / f"results_{ts}.csv"
    df = pd.DataFrame([asdict(s) for s in all_stats])
    df.to_csv(csv_path, index=False)

    console.print(f"\n[green]Results saved to {json_path} and {csv_path}[/green]")


def plot_results(all_stats: list[BenchmarkStats]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        engines = list(dict.fromkeys(s.engine for s in all_stats))
        concs = sorted(set(s.concurrency for s in all_stats))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Benchmark — qwen2.5:7b", fontsize=13)

        metrics = [
            ("throughput_tok_s", "Throughput (tok/s)", axes[0]),
            ("ttft_p50_ms",      "TTFT p50 (ms)",       axes[1]),
            ("latency_p95_ms",   "Latency p95 (ms)",    axes[2]),
        ]

        x = np.arange(len(concs))
        width = 0.8 / len(engines)

        for ax_idx, (metric, ylabel, ax) in enumerate(metrics):
            for i, eng in enumerate(engines):
                vals = []
                for c in concs:
                    match = next((s for s in all_stats if s.engine == eng and s.concurrency == c), None)
                    vals.append(getattr(match, metric, 0) if match else 0)
                offset = (i - len(engines) / 2 + 0.5) * width
                ax.bar(x + offset, vals, width * 0.9, label=eng)

            ax.set_xticks(x)
            ax.set_xticklabels([f"c={c}" for c in concs])
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            if ax_idx == 0:
                ax.legend()

        plt.tight_layout()
        ts = int(time.time())
        path = OUTPUT_DIR / f"benchmark_{ts}.png"
        plt.savefig(path, dpi=150)
        console.print(f"[green]Chart saved to {path}[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not generate chart: {e}[/yellow]")


# ── Entry point ───────────────────────────────────────────────────────────────

def load_prompts() -> list[str]:
    pf = Path("/app/prompts.json")
    if pf.exists():
        return json.loads(pf.read_text())
    # default prompts
    return [
        "Explain the theory of general relativity in simple terms.",
        "Write a Python function to compute Fibonacci numbers iteratively.",
        "What are the main differences between Docker and Kubernetes?",
        "Summarize the plot of 'One Hundred Years of Solitude'.",
        "Describe the steps to deploy a web app on AWS.",
        "What is the difference between supervised and unsupervised learning?",
        "Write a haiku about machine learning.",
        "Explain how attention mechanisms work in transformers.",
    ]


async def main() -> None:
    prompts = load_prompts()
    adapters: list[EngineAdapter] = [
        AirLLMAdapter(),
        LlamaCppAdapter(),
        OllamaAdapter(),
        # VLLMAdapter(),
    ]

    all_stats: list[BenchmarkStats] = []

    console.rule("[bold cyan]LLM Inference Benchmark[/bold cyan]")
    console.print(f"Model      : [yellow]{MODEL_NAME}[/yellow]")
    console.print(f"Max tokens : {NUM_TOKENS}")
    console.print(f"Concurrency: {CONCURRENCY_LEVELS}")
    console.print(f"Prompts    : {len(prompts)}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total_tasks = len(adapters) * len(CONCURRENCY_LEVELS)
        task = progress.add_task("Benchmarking...", total=total_tasks)

        for adapter in adapters:
            for concurrency in CONCURRENCY_LEVELS:
                progress.update(task, description=f"{adapter.name} @ c={concurrency}")
                stats = await bench_engine(
                    adapter, prompts, concurrency, NUM_TOKENS, WARMUP
                )
                all_stats.append(stats)
                progress.advance(task)

    print_table(all_stats)
    save_results(all_stats)
    plot_results(all_stats)


if __name__ == "__main__":
    asyncio.run(main())
