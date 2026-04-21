FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    airllm==0.2.9 \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    torch==2.2.2 \
    transformers==4.40.2

WORKDIR /app
COPY server.py .

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
