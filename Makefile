# ──────────────────────────────────────────────────────────────────────────────
#  LLM Inference Benchmark — Makefile
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help build run run-monitoring run-dashboard stop clean logs pull-model

DOCKER_COMPOSE = docker compose
ENV_FILE = .env

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all engine images
	$(DOCKER_COMPOSE) build

run: ## Run the benchmark (engines + orchestrator)
	@cp -n .env.example $(ENV_FILE) 2>/dev/null || true
	$(DOCKER_COMPOSE) --env-file $(ENV_FILE) up --abort-on-container-exit orchestrator

run-dashboard: ## Start the supervision dashboard (http://localhost:8090)
	@cp -n .env.example $(ENV_FILE) 2>/dev/null || true
	$(DOCKER_COMPOSE) --env-file $(ENV_FILE) up dashboard -d
	@echo "Dashboard disponible sur http://localhost:8090"

run-monitoring: ## Run benchmark + Prometheus + Grafana
	@cp -n .env.example $(ENV_FILE) 2>/dev/null || true
	$(DOCKER_COMPOSE) --env-file $(ENV_FILE) --profile monitoring up --abort-on-container-exit orchestrator

stop: ## Stop and remove all containers
	$(DOCKER_COMPOSE) down --remove-orphans

clean: ## Remove containers, images, and result files
	$(DOCKER_COMPOSE) down --rmi local --volumes --remove-orphans
	rm -rf results/*

logs: ## Tail logs of all running containers
	$(DOCKER_COMPOSE) logs -f

logs-engine: ## Tail logs of a specific engine (e.g. make logs-engine ENGINE=vllm)
	$(DOCKER_COMPOSE) logs -f $(ENGINE)

# ── Model helpers ──────────────────────────────────────────────────────────────

pull-ollama: ## Pull model into Ollama (requires running Ollama container)
	docker exec bench-ollama ollama pull $(MODEL_NAME)

download-gguf: ## Download a GGUF model with huggingface-cli
	@echo "Downloading $(GGUF_URL) to ./models/"
	mkdir -p models
	curl -L "$(GGUF_URL)" -o "models/$(GGUF_MODEL)"

# ── Quick test ─────────────────────────────────────────────────────────────────

test-health: ## Check health endpoints of all engines
	@echo "=== AirLLM ===" && curl -sf http://localhost:8081/health && echo
	@echo "=== llama.cpp ===" && curl -sf http://localhost:8082/health && echo
	@echo "=== Ollama ===" && curl -sf http://localhost:11434/api/tags && echo
	@echo "=== vLLM ===" && curl -sf http://localhost:8000/health && echo
