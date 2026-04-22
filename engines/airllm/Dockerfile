# Part de l'image PyTorch officielle — torch déjà installé, pas de re-téléchargement CUDA
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    airllm==2.11.0 \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    torch==2.2.2 \
    transformers==4.40.2 \
    optimum==1.17.1

WORKDIR /app
COPY server.py .

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
