FROM vllm/vllm-openai:latest

# vLLM already ships with OpenAI-compatible /v1/completions endpoint.
# Nothing extra needed — entrypoint is set in docker-compose command.
# This file exists to allow custom patches or extra packages if needed.

# Example: add a custom health-check shim
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=30 \
    CMD curl -sf http://localhost:8000/health || exit 1
