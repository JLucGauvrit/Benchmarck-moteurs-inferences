FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl && \
    rm -rf /var/lib/apt/lists/*

# Build llama.cpp with CUDA backend
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /llama.cpp
WORKDIR /llama.cpp
RUN cmake -B build \
    -DLLAMA_CUDA=ON \
    -DLLAMA_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" && \
    cmake --build build --config Release -j$(nproc) --target llama-server

# ── Runtime image ──────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

ENV MODEL_PATH=/models/model.gguf
ENV PORT=8080
ENV N_GPU_LAYERS=99
ENV CTX_SIZE=4096
ENV N_PARALLEL=4

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=20 \
    CMD curl -sf http://localhost:${PORT}/health

CMD llama-server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --ctx-size "${CTX_SIZE}" \
    --parallel "${N_PARALLEL}" \
    --flash-attn \
    --no-mmap
