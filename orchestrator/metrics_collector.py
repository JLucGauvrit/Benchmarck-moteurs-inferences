"""
Collecte en arrière-plan : GPU VRAM, GPU util%, CPU%, RAM
Nécessite : pip install pynvml psutil
"""
import asyncio
import time
import json
from pathlib import Path

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

import psutil

class MetricsCollector:
    def __init__(self, output_dir: Path, interval: float = 1.0):
        self.output_dir = output_dir
        self.interval = interval
        self.records: list[dict] = []
        self._running = False
        self._task = None
        self.label = ""

    def start(self, label: str):
        self.label = label
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> list[dict]:
        self._running = False
        if self._task:
            await self._task
        return self.records

    async def _loop(self):
        while self._running:
            rec = {
                "ts": time.time(),
                "label": self.label,
                "cpu_pct": psutil.cpu_percent(),
                "ram_used_gb": psutil.virtual_memory().used / 1e9,
                "ram_total_gb": psutil.virtual_memory().total / 1e9,
                "gpus": [],
            }
            if NVML_OK:
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    rec["gpus"].append({
                        "id": i,
                        "name": pynvml.nvmlDeviceGetName(h),
                        "vram_used_mb": mem.used // 1024 // 1024,
                        "vram_total_mb": mem.total // 1024 // 1024,
                        "gpu_util_pct": util.gpu,
                        "mem_util_pct": util.memory,
                    })
            self.records.append(rec)
            await asyncio.sleep(self.interval)

    def save(self, engine: str, concurrency: int):
        path = self.output_dir / f"metrics_{engine}_c{concurrency}_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump(self.records, f)
        self.records = []
        