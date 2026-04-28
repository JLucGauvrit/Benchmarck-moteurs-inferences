import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import torch
from airllm import AutoModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen/Qwen2.5-7B-Instruct")
COMPRESSION = os.getenv("COMPRESSION", "4bit")
CACHE_PATH = os.getenv("AIRLLM_CACHE", "/cache/airllm")

_model = None


def _load_model():
    global _model
    _model = AutoModel.from_pretrained(
        MODEL_PATH,
        compression=COMPRESSION if COMPRESSION != "None" else None,
        layer_shards_saving_path=CACHE_PATH,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)
    yield


app = FastAPI(title="AirLLM benchmark server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 1.0


@app.get("/health")
async def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    try:
        inputs = _model.tokenizer(req.prompt, return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = _model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
        )

        text = _model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        payload = {
            "text": text,
            "latency_ms": latency_ms,
            "tokens_out": len(outputs[0]),
        }

        return StreamingResponse(
            iter([f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"]),
            media_type="text/event-stream",
        )

    except Exception as e:
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': str(e)})}\n\n"]),
            media_type="text/event-stream",
        )
