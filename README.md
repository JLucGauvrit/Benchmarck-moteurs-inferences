# LLM Inference Engine Benchmark

Benchmark Docker-based pour comparer **AirLLM**, **llama.cpp**, **Ollama** et **vLLM** sur GPU.

## Métriques mesurées

| Métrique | Description |
|---|---|
| **TTFT** | Time-to-First-Token (ms) |
| **Throughput** | Tokens générés / seconde |
| **Latence p50/p95/p99** | Percentiles de latence totale |
| **Taux de succès** | Requêtes réussies / total |

## Prérequis

- Docker Engine ≥ 24 + Docker Compose v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- GPU NVIDIA avec drivers ≥ 525
- ~50 GB d'espace disque pour les modèles

## Structure

```
llm-benchmark/
├── docker-compose.yml          # Orchestration principale
├── .env.example                # Variables de configuration
├── Makefile                    # Commandes utilitaires
├── prometheus.yml              # Config scraping métriques GPU
├── orchestrator/
│   ├── Dockerfile
│   ├── runner.py               # Logique de benchmark async
│   └── prompts.json            # Prompts de test
├── engines/
│   ├── airllm/                 # Layer-splitting CPU↔GPU
│   │   ├── Dockerfile
│   │   └── server.py           # FastAPI + SSE streaming
│   ├── llamacpp/               # GGUF + CUDA offload
│   │   └── Dockerfile          # Build llama-server CUDA
│   ├── ollama/                 # REST wrapper llama.cpp
│   │   └── Dockerfile
│   └── vllm/                   # PagedAttention
│       └── Dockerfile
└── results/                    # JSON + CSV + PNG générés
```

## Démarrage rapide

### 1. Configurer l'environnement

```bash
cp .env.example .env
# Éditez .env selon votre setup
```

### 2. Préparer les modèles

**Pour vLLM et AirLLM** (modèle HuggingFace) :
```bash
mkdir -p models
# Exemple avec huggingface-cli
pip install huggingface_hub
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistralai/Mistral-7B-v0.1
```

**Pour llama.cpp** (format GGUF) :
```bash
# Télécharger depuis HuggingFace Hub
curl -L "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf" \
     -o models/mistral-7b-q4_k_m.gguf
```

**Pour Ollama** :
```bash
# Sera téléchargé automatiquement au premier pull
make pull-ollama MODEL_NAME=mistral
```

### 3. Builder les images

```bash
make build
```

### 4. Lancer le benchmark

```bash
make run
```

Les résultats sont générés dans `./results/` :
- `results_<ts>.json` — données brutes
- `results_<ts>.csv` — tableau pour analyse
- `benchmark_<ts>.png` — graphiques comparatifs

### 5. Monitoring GPU (optionnel)

```bash
make run-monitoring
# Grafana accessible sur http://localhost:3000 (admin/admin)
```

## Configuration avancée

### Variables `.env`

| Variable | Défaut | Description |
|---|---|---|
| `MODEL_NAME` | `mistralai/Mistral-7B-v0.1` | Nom du modèle HF ou dossier local |
| `GGUF_MODEL` | `mistral-7b-q4_k_m.gguf` | Fichier GGUF pour llama.cpp |
| `N_GPU_LAYERS` | `99` | Layers GPU pour llama.cpp (99 = tout) |
| `TENSOR_PARALLEL` | `1` | Nombre de GPUs pour vLLM |
| `CTX_SIZE` | `4096` | Taille de contexte maximale |
| `CONCURRENCY` | `1,4,8` | Niveaux de concurrence testés |
| `NUM_TOKENS` | `256` | Tokens maximum à générer |
| `WARMUP_REQUESTS` | `3` | Requêtes de chauffe avant mesure |

### Multi-GPU

Pour vLLM avec 2 GPUs :
```bash
TENSOR_PARALLEL=2 make run
```

### Ajouter des prompts personnalisés

Modifiez `orchestrator/prompts.json` :
```json
["Votre prompt 1", "Votre prompt 2", ...]
```

## Comparaison des moteurs

| Moteur | Avantages | Cas d'usage idéal |
|---|---|---|
| **AirLLM** | Gros modèles sur peu de VRAM (layer splitting) | Modèles 70B+ sur GPU 24GB |
| **llama.cpp** | Léger, GGUF quantifié, CPU+GPU hybride | Déploiement local, machines limitées |
| **Ollama** | Simple, API REST propre, gestion auto des modèles | Prototypage rapide, dev local |
| **vLLM** | Throughput maximal (PagedAttention), batch continu | Production, haute concurrence |

## Résultats typiques (Mistral-7B, RTX 3090)

| Moteur | Throughput (tok/s) | TTFT p50 (ms) |
|---|---|---|
| vLLM | ~1800 | ~45 |
| llama.cpp | ~900 | ~120 |
| Ollama | ~850 | ~130 |
| AirLLM | ~120 | ~800 |

> AirLLM est intentionnellement plus lent — son avantage est de faire tourner des modèles qui ne tiennent pas en VRAM.
