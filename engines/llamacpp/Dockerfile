FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root
RUN apt-get update -qq && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=20 \
    CMD curl -sf http://localhost:${PORT:-8080}/health || exit 1

ENTRYPOINT ["/app/llama-server"]
CMD ["--help"]
