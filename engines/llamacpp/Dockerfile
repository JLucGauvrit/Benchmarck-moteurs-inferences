FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root
RUN apt-get update -qq && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV MODEL_PATH=/models/model.gguf
ENV PORT=8080
ENV N_GPU_LAYERS=99
ENV CTX_SIZE=4096
ENV N_PARALLEL=4

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=20 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

ENTRYPOINT []
CMD ["/bin/sh", "-c", \
  "/app/llama-server \
   --model \"${MODEL_PATH}\" \
   --port \"${PORT}\" \
   --host 0.0.0.0 \
   --n-gpu-layers \"${N_GPU_LAYERS}\" \
   --ctx-size \"${CTX_SIZE}\" \
   --parallel \"${N_PARALLEL}\""]
