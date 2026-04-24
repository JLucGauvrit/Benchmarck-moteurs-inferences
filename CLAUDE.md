# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Docker-based benchmark suite comparing LLM inference engines: **AirLLM**, **llama.cpp**, **Ollama**, and **vLLM** (disabled by default). The orchestrator runs async HTTP requests against each engine and measures TTFT, throughput (tok/s), and latency percentiles (p50/p95/p99).

## Common Commands

```bash
# Setup
cp .env.example .env          # configure before first run

# Build & run
make build                    # build all engine Docker images
make run                      # run benchmark (engines + orchestrator)
make run-monitoring           # benchmark + Prometheus + Grafana (http://localhost:3000)

# Utilities
make stop                     # stop and remove containers
make clean                    # stop + remove images + delete results/
make logs                     # tail all container logs
make logs-engine ENGINE=airllm  # tail a specific engine
make test-health              # curl all engine /health endpoints

# Model management
make pull-ollama MODEL_NAME=qwen2.5:7b      # pull model into running Ollama container
make download-gguf GGUF_URL=<url> GGUF_MODEL=<filename>  # download GGUF file to ./models/
```

## Architecture

```
orchestrator/runner.py        # async benchmark loop (httpx + asyncio.Semaphore)
orchestrator/metrics_collector.py  # background GPU/CPU/RAM sampling via pynvml + psutil
orchestrator/prompts.json     # 16 test prompts used for all engines
engines/airllm/server.py      # FastAPI + SSE streaming server wrapping AirLLM
engines/airllm/Dockerfile     # pytorch/pytorch:2.2.2-cuda12.1 base
engines/llamacpp/Dockerfile   # ghcr.io/ggml-org/llama.cpp:server-cuda base
docker-compose.yml            # all services, healthchecks, GPU passthrough
prometheus.yml                # GPU metrics scraping config (dcgm-exporter + cadvisor)
results/                      # output: results_<ts>.json, results_<ts>.csv, benchmark_<ts>.png
```

### Orchestrator flow

`runner.py` defines one `EngineAdapter` subclass per engine, each implementing `async generate(client, prompt, max_tokens) -> RequestResult`. The `main()` function runs `bench_engine()` for every `(adapter, concurrency_level)` pair, measures wall-clock TTFT and total latency per streaming token, then calls `compute_stats()` to aggregate into `BenchmarkStats`.

### Engine API contracts

| Engine | Endpoint | Protocol |
|--------|----------|----------|
| AirLLM | `POST /generate` | SSE, `data: {token, token_count, elapsed_ms}` then `data: [DONE]` |
| llama.cpp | `POST /completion` | SSE, `data: {stop: bool}` per token |
| Ollama | `POST /api/generate` | NDJSON, `{done: bool}` per line |
| vLLM | `POST /v1/completions` | SSE, OpenAI-compatible, `data: [DONE]` |

### Enabling/disabling engines

- **vLLM** is commented out in `docker-compose.yml` (`depends_on`) and in `runner.py` (`adapters` list). Uncomment both to enable it.
- **Monitoring stack** (Prometheus, Grafana, dcgm-exporter, cadvisor) uses Docker Compose `profiles: [monitoring]` — only activated by `make run-monitoring`.

## Key Configuration (`.env`)

| Variable | Default | Notes |
|----------|---------|-------|
| `MODEL_NAME` | `mistralai/Mistral-7B-v0.1` | HuggingFace repo ID or local folder under `./models/` |
| `GGUF_MODEL` | `mistral-7b-q4_k_m.gguf` | Filename inside `./models/` for llama.cpp |
| `CONCURRENCY` | `1,4,8` | Comma-separated levels; orchestrator tests each |
| `NUM_TOKENS` | `256` | Max tokens generated per request |
| `WARMUP_REQUESTS` | `3` | Discarded warm-up requests before measurement |
| `N_GPU_LAYERS` | `99` | llama.cpp GPU layer offload (99 = all) |
| `AIRLLM_COMPRESSION` | `4bit` | AirLLM quantisation (`4bit`, `8bit`, or `None`) |

## Model paths

- **HuggingFace models** (AirLLM, vLLM): place under `./models/<org>/<model>/` — mounted read-only at `/models`
- **GGUF models** (llama.cpp): place directly in `./models/` (e.g. `./models/qwen2.5-7b-instruct-q3_k_m.gguf`)
- **Ollama models**: stored in `./models/ollama/` — pulled via `make pull-ollama`
- **AirLLM split cache**: stored in `./cache/airllm/` (layer shards created on first load)

Both `models/` and `cache/` are git-ignored.

## Adding a new engine

1. Add a `Dockerfile` under `engines/<name>/`
2. Add the service to `docker-compose.yml` with a `/health` endpoint and healthcheck
3. Subclass `EngineAdapter` in `orchestrator/runner.py` and implement `generate()`
4. Add the adapter instance to the `adapters` list in `main()`
5. Add the service URL env var to the orchestrator service in `docker-compose.yml`
