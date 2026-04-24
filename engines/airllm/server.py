import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager

import torch
from airllm import AutoModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer

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


app = FastAPI(title="AirLLM inference server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 1.0
    stream: bool = True


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/generate")
async def generate(req: GenerateRequest):
    model = _model

    async def token_stream():
        inputs = model.tokenizer(req.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        streamer = TextIteratorStreamer(model.tokenizer, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": req.max_new_tokens,
            "streamer": streamer,
            "do_sample": req.temperature > 0,
        }
        if req.temperature > 0:
            generation_kwargs["temperature"] = req.temperature

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        token_count = 0
        t0 = time.perf_counter()
        for text_piece in streamer:
            token_count += 1
            yield (
                f"data: {json.dumps({'token': text_piece, 'token_count': token_count, 'elapsed_ms': round((time.perf_counter() - t0) * 1000, 1)})}\n\n"
            )
            await asyncio.sleep(0)

        thread.join()
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
