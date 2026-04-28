"""LLM Inference Benchmark Orchestrator"""

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
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

# ── CONFIG ────────────────────────────────────────────────────

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/results"))
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")
NUM_TOKENS = int(os.getenv("NUM_TOKENS", "256"))
WARMUP = int(os.getenv("WARMUP_REQUESTS", "1"))
CONCURRENCY_LEVELS = [int(x) for x in os.getenv("CONCURRENCY", "1,4,8").split(",")]
ENGINES_FILTER = (
    set(os.getenv("ENGINES_FILTER", "").split(","))
    if os.getenv("ENGINES_FILTER") else None
)
PROMPTS_FILE = Path(os.getenv("BENCH_PROMPTS_FILE", "/app/prompts.json"))

# ── DATA MODEL ────────────────────────────────────────────────

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


# ── ENGINE ADAPTERS ───────────────────────────────────────────

class EngineAdapter:
    name: str
    base_url: str
    request_timeout: float = 120.0
    warmup_timeout: float = 60.0
    max_tokens_override: Optional[int] = None

    async def generate(self, client: httpx.AsyncClient, prompt: str, max_tokens: int) -> RequestResult:
        raise NotImplementedError


class AirLLMAdapter(EngineAdapter):
    name = "airllm"
    base_url = os.getenv("AIRLLM_URL", "http://airllm:8080")
    request_timeout = 600.0
    warmup_timeout = 600.0
    max_tokens_override = 32  # layer-splitting is very slow

    async def generate(self, client: httpx.AsyncClient, prompt: str, max_tokens: int) -> RequestResult:
        t0 = time.perf_counter()
        try:
            async with client.stream(
                "POST", f"{self.base_url}/generate",
                json={"prompt": prompt, "max_new_tokens": max_tokens},
                timeout=httpx.Timeout(self.request_timeout),
            ) as resp:
                resp.raise_for_status()
                ttft_ms: Optional[float] = None
                tokens_out = 0
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data:"):
                        continue
                    payload = raw[5:].strip()
                    if payload == "[DONE]":
                        break
                    data = json.loads(payload)
                    if "error" in data:
                        raise RuntimeError(data["error"])
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    tokens_out = data.get("tokens_out", tokens_out)

            total_ms = (time.perf_counter() - t0) * 1000
            return RequestResult(
                engine=self.name, concurrency=0, success=True,
                ttft_ms=ttft_ms or total_ms,
                total_latency_ms=total_ms,
                output_tokens=tokens_out,
            )
        except Exception as e:
            return RequestResult(
                engine=self.name, concurrency=0, success=False,
                ttft_ms=0, total_latency_ms=0, output_tokens=0, error=str(e),
            )


class LlamaCppAdapter(EngineAdapter):
    name = "llamacpp"
    base_url = os.getenv("LLAMACPP_URL", "http://llamacpp:8080")
    request_timeout = 180.0

    async def generate(self, client: httpx.AsyncClient, prompt: str, max_tokens: int) -> RequestResult:
        t0 = time.perf_counter()
        try:
            async with client.stream(
                "POST", f"{self.base_url}/completion",
                json={"prompt": prompt, "n_predict": max_tokens, "stream": True},
                timeout=httpx.Timeout(self.request_timeout),
            ) as resp:
                resp.raise_for_status()
                ttft_ms: Optional[float] = None
                tokens_out = 0
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data:"):
                        continue
                    data = json.loads(raw[5:].strip())
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    tokens_out += 1
                    if data.get("stop"):
                        break

            total_ms = (time.perf_counter() - t0) * 1000
            return RequestResult(
                engine=self.name, concurrency=0, success=True,
                ttft_ms=ttft_ms or total_ms,
                total_latency_ms=total_ms,
                output_tokens=tokens_out,
            )
        except Exception as e:
            return RequestResult(
                engine=self.name, concurrency=0, success=False,
                ttft_ms=0, total_latency_ms=0, output_tokens=0, error=str(e),
            )


class OllamaAdapter(EngineAdapter):
    name = "ollama"
    base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    request_timeout = 180.0
    _model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    async def generate(self, client: httpx.AsyncClient, prompt: str, max_tokens: int) -> RequestResult:
        t0 = time.perf_counter()
        try:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "options": {"num_predict": max_tokens}},
                timeout=httpx.Timeout(self.request_timeout),
            ) as resp:
                resp.raise_for_status()
                ttft_ms: Optional[float] = None
                tokens_out = 0
                async for raw in resp.aiter_lines():
                    if not raw:
                        continue
                    data = json.loads(raw)
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    if data.get("done"):
                        tokens_out = data.get("eval_count", tokens_out)
                        break
                    tokens_out += 1

            total_ms = (time.perf_counter() - t0) * 1000
            return RequestResult(
                engine=self.name, concurrency=0, success=True,
                ttft_ms=ttft_ms or total_ms,
                total_latency_ms=total_ms,
                output_tokens=tokens_out,
            )
        except Exception as e:
            return RequestResult(
                engine=self.name, concurrency=0, success=False,
                ttft_ms=0, total_latency_ms=0, output_tokens=0, error=str(e),
            )


class VLLMAdapter(EngineAdapter):
    name = "vllm"
    base_url = os.getenv("VLLM_URL", "http://vllm:8000")
    request_timeout = 180.0
    _model = "/models/Qwen/Qwen2.5-7B-Instruct-AWQ"

    async def generate(self, client: httpx.AsyncClient, prompt: str, max_tokens: int) -> RequestResult:
        t0 = time.perf_counter()
        try:
            async with client.stream(
                "POST", f"{self.base_url}/v1/completions",
                json={"model": self._model, "prompt": prompt, "max_tokens": max_tokens, "stream": True},
                timeout=httpx.Timeout(self.request_timeout),
            ) as resp:
                resp.raise_for_status()
                ttft_ms: Optional[float] = None
                tokens_out = 0
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data:"):
                        continue
                    payload = raw[5:].strip()
                    if payload == "[DONE]":
                        break
                    data = json.loads(payload)
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    tokens_out += len(data.get("choices", []))

            total_ms = (time.perf_counter() - t0) * 1000
            return RequestResult(
                engine=self.name, concurrency=0, success=True,
                ttft_ms=ttft_ms or total_ms,
                total_latency_ms=total_ms,
                output_tokens=tokens_out,
            )
        except Exception as e:
            return RequestResult(
                engine=self.name, concurrency=0, success=False,
                ttft_ms=0, total_latency_ms=0, output_tokens=0, error=str(e),
            )


# ── SAFE CALL ────────────────────────────────────────────────

async def safe_call(adapter: EngineAdapter, client, prompt, max_tokens):
    try:
        return await asyncio.wait_for(
            adapter.generate(client, prompt, max_tokens),
            timeout=adapter.request_timeout,
        )
    except asyncio.TimeoutError:
        return RequestResult(
            engine=adapter.name, concurrency=0, success=False,
            ttft_ms=0, total_latency_ms=adapter.request_timeout * 1000,
            output_tokens=0, error="timeout",
        )
    except Exception as e:
        return RequestResult(
            engine=adapter.name, concurrency=0, success=False,
            ttft_ms=0, total_latency_ms=0, output_tokens=0, error=str(e),
        )


# ── CORE BENCH ───────────────────────────────────────────────

async def bench_engine(adapter: EngineAdapter, prompts, concurrency, max_tokens, warmup):
    async with httpx.AsyncClient(timeout=httpx.Timeout(adapter.request_timeout)) as client:

        for i in range(warmup):
            try:
                await asyncio.wait_for(
                    adapter.generate(client, prompts[i % len(prompts)], max_tokens),
                    timeout=adapter.warmup_timeout,
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

        tasks = [run_one(prompts[i % len(prompts)]) for i in range(len(prompts))]
        await asyncio.gather(*tasks, return_exceptions=True)

    return compute_stats(results, adapter.name, concurrency)


# ── STATS ────────────────────────────────────────────────────

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


# ── HELPERS ──────────────────────────────────────────────────

def load_prompts(n: int = 4) -> list[str]:
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE) as f:
            prompts = json.load(f)
        return prompts[:n]
    return [
        "Explain transformers in simple terms.",
        "Write a Python quicksort.",
        "What is Kubernetes?",
        "Summarize machine learning.",
    ]


def print_table(stats: list[BenchmarkStats]):
    table = Table(title="Benchmark Results")
    table.add_column("Engine")
    table.add_column("c")
    table.add_column("Success")
    table.add_column("Tok/s")
    table.add_column("TTFT p50")
    table.add_column("TTFT p95")
    table.add_column("Lat p95")

    for s in stats:
        table.add_row(
            s.engine, str(s.concurrency),
            f"{s.n_success}/{s.n_requests}",
            str(s.throughput_tok_s),
            str(round(s.ttft_p50_ms, 1)),
            str(round(s.ttft_p95_ms, 1)),
            str(round(s.latency_p95_ms, 1)),
        )
    console.print(table)


def save_results(stats: list[BenchmarkStats]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = OUTPUT_DIR / f"results_{ts}.json"
    with open(path, "w") as f:
        json.dump([asdict(s) for s in stats], f, indent=2)
    console.print(f"[green]Saved → {path}[/green]")


# ── ENTRY ────────────────────────────────────────────────────

async def main():
    console.print("[bold cyan]LLM Benchmark[/bold cyan]")
    console.print(f"Model: {MODEL_NAME}  |  Tokens: {NUM_TOKENS}  |  Concurrency: {CONCURRENCY_LEVELS}")
    console.print()

    all_adapters: list[EngineAdapter] = [
        AirLLMAdapter(),
        LlamaCppAdapter(),
        OllamaAdapter(),
        VLLMAdapter(),
    ]

    adapters = (
        [a for a in all_adapters if a.name in ENGINES_FILTER]
        if ENGINES_FILTER else all_adapters
    )

    prompts = load_prompts(n=4)
    all_stats: list[BenchmarkStats] = []

    for adapter in adapters:
        # AirLLM: single concurrency level only (layer-splitting is sequential anyway)
        levels = [1] if adapter.name == "airllm" else CONCURRENCY_LEVELS
        max_toks = adapter.max_tokens_override or NUM_TOKENS
        warmup_count = 0 if adapter.name == "airllm" else WARMUP

        console.print(f"\n[bold]{adapter.name}[/bold] — {len(prompts)} prompts × {levels} concurrency, {max_toks} tokens")

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"{adapter.name}", total=len(levels))

            for c in levels:
                progress.update(task, description=f"{adapter.name} @ c={c}")
                stats = await bench_engine(adapter, prompts, c, max_toks, warmup_count)
                all_stats.append(stats)
                success_rate = f"{stats.n_success}/{stats.n_requests}"
                progress.update(task, description=f"{adapter.name} @ c={c} → {success_rate} ok, {stats.throughput_tok_s} tok/s")
                progress.advance(task)

    print_table(all_stats)
    save_results(all_stats)


if __name__ == "__main__":
    asyncio.run(main())
