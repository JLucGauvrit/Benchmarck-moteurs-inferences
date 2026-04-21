"""
AirLLM inference server — streams tokens via SSE.
AirLLM splits model layers across CPU/GPU, enabling large models on limited VRAM.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import torch
from airllm import AutoModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="AirLLM inference server")

MODEL_PATH = os.getenv("MODEL_PATH", "/models/mistral")
_model = None


def get_model():
    global _model
    if _model is None:
        compression = os.getenv("COMPRESSION", "4bit")  # 4bit | None
        _model = AutoModel.from_pretrained(
            MODEL_PATH,
            compression=compression if compression != "None" else None,
        )
    return _model


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
    model = get_model()

    async def token_stream():
        input_text = req.prompt
        # AirLLM tokenizes internally; we generate token-by-token
        inputs = model.tokenizer(input_text, return_tensors="pt").to(model.device)
        generated = inputs["input_ids"]

        t0 = time.perf_counter()
        token_count = 0

        for _ in range(req.max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :]
            if req.temperature > 0:
                probs = torch.softmax(logits / req.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            token_count += 1

            text_piece = model.tokenizer.decode(next_token[0], skip_special_tokens=True)
            chunk = {
                "token": text_piece,
                "token_count": token_count,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            if next_token.item() == model.tokenizer.eos_token_id:
                break

            await asyncio.sleep(0)  # yield control

        yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(token_stream(), media_type="text/event-stream")

    # non-streaming fallback
    inputs = model.tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    text = model.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"text": text}
