"""FastAPI backend for LLM benchmark supervision dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

RESULTS_DIR = Path(os.getenv("OUTPUT_DIR", "/results"))
STATUS_FILE = RESULTS_DIR / "status.json"
STREAM_FILE = RESULTS_DIR / "stream.jsonl"

ENGINE_URLS = {
    "airllm":   os.getenv("AIRLLM_URL",   "http://airllm:8080"),
    "llamacpp": os.getenv("LLAMACPP_URL", "http://llamacpp:8080"),
    "ollama":   os.getenv("OLLAMA_URL",   "http://ollama:11434"),
}

_process: Optional[asyncio.subprocess.Process] = None
_stream_offset: int = 0

# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/benchmark/start")
async def start_benchmark():
    global _process, _stream_offset
    if _process and _process.returncode is None:
        return JSONResponse({"error": "benchmark already running"}, status_code=400)

    _stream_offset = 0
    env = {**os.environ, "OUTPUT_DIR": str(RESULTS_DIR)}

    _process = await asyncio.create_subprocess_exec(
        sys.executable, "/app/runner.py",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    return {"started": True, "pid": _process.pid}


@app.post("/api/benchmark/stop")
async def stop_benchmark():
    global _process
    if _process and _process.returncode is None:
        _process.terminate()
        try:
            await asyncio.wait_for(_process.wait(), timeout=5)
        except asyncio.TimeoutError:
            _process.kill()
        return {"stopped": True}
    return {"stopped": False}


@app.get("/api/benchmark/status")
async def benchmark_status():
    if STATUS_FILE.exists():
        try:
            return JSONResponse(json.loads(STATUS_FILE.read_text()))
        except Exception:
            pass
    return {"state": "idle"}


@app.get("/api/engines/health")
async def engines_health():
    return await _check_engines()


@app.get("/api/results")
async def list_results():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RESULTS_DIR.glob("results_*.json"), reverse=True)
    return [f.name for f in files]


@app.get("/api/results/{filename}")
async def get_result(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists() or not path.name.startswith("results_"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(json.loads(path.read_text()))


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _check_engines() -> dict:
    results = {}
    async with httpx.AsyncClient(timeout=2) as client:
        for name, base_url in ENGINE_URLS.items():
            url = f"{base_url}/api/tags" if name == "ollama" else f"{base_url}/health"
            try:
                resp = await client.get(url)
                results[name] = resp.status_code < 400
            except Exception:
                results[name] = False
    return results


def _collect_metrics() -> dict:
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    gpus = []
    if NVML_OK:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                gpus.append({
                    "id": i,
                    "name": name,
                    "gpu_util_pct": util.gpu,
                    "vram_used_mb": mem_info.used // (1024 * 1024),
                    "vram_total_mb": mem_info.total // (1024 * 1024),
                })
        except Exception:
            pass
    return {
        "cpu_pct": cpu,
        "ram_used_gb": round(mem.used / 1e9, 2),
        "ram_total_gb": round(mem.total / 1e9, 2),
        "gpus": gpus,
    }


def _read_new_stream_lines() -> list:
    global _stream_offset
    if not STREAM_FILE.exists():
        return []
    lines = []
    try:
        with open(STREAM_FILE, "r") as f:
            f.seek(_stream_offset)
            for raw in f:
                raw = raw.strip()
                if raw:
                    try:
                        lines.append(json.loads(raw))
                    except Exception:
                        pass
            _stream_offset = f.tell()
    except Exception:
        pass
    return lines


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    psutil.cpu_percent(interval=None)  # prime the cpu sampler
    tick = 0
    try:
        while True:
            metrics = _collect_metrics()

            status: dict = {"state": "idle"}
            if STATUS_FILE.exists():
                try:
                    status = json.loads(STATUS_FILE.read_text())
                except Exception:
                    pass

            new_lines = _read_new_stream_lines()

            payload: dict = {
                "type": "tick",
                "ts": time.time(),
                "metrics": metrics,
                "status": status,
                "stream": new_lines,
            }

            if tick % 5 == 0:
                payload["health"] = await _check_engines()

            tick += 1
            await ws.send_json(payload)
            await asyncio.sleep(1)
    except (WebSocketDisconnect, Exception):
        pass
