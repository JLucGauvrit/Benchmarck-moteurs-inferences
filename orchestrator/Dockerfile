FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    aiohttp==3.9.5 \
    httpx==0.27.0 \
    rich==13.7.1 \
    pandas==2.2.2 \
    matplotlib==3.9.0 \
    numpy==1.26.4 \
    pydantic==2.7.1 \
    pynvml==11.5.0 \
    psutil==5.9.8

COPY . .

ENTRYPOINT ["python", "runner.py"]
