---
title: Model Speed Comparator
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Model Speed Comparator

Compare PyTorch baseline vs ONNX vs INT8 Quantized inference — same model, same prediction, dramatically different performance.

Built to demonstrate real-world AI inference optimization techniques used in production ML systems and AI accelerator pipelines.

## What It Does

Takes any text input and runs it through 3 versions of the same NLP model (DistilBERT sentiment classifier):

| Variant | Format | What changes |
|---|---|---|
| Baseline | PyTorch .bin | Standard HuggingFace model, no optimization |
| ONNX | .onnx | Exported + graph-optimized by ONNX Runtime |
| Quantized | INT8 .onnx | Weights compressed from FP32 to INT8 |

## Key Results (CPU)

| Variant | Latency | Size | vs Baseline |
|---|---|---|---|
| PyTorch Baseline | 5594ms | 268MB | 1x |
| ONNX | 547ms | 255MB | 10x faster |
| INT8 Quantized | 26ms | 64MB | 213x faster, 4x smaller |

## Setup

```bash
git clone https://github.com/Mridul0603/Model-Speed-Comparator
cd Model-Speed-Comparator
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000

## Tech Stack

- FastAPI
- HuggingFace Transformers
- ONNX Runtime
- Optimum
- Docker

## API

POST /compare - runs all 3 variants and returns latency comparison
POST /benchmark - runs 20x stress test with p95 stats
GET /history - last 10 comparisons
GET /stats - session aggregate stats
GET /health - health check
