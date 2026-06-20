# ⚡ Model Speed Comparator

> Compare PyTorch baseline vs ONNX vs INT8 Quantized inference — same model, same prediction, dramatically different performance.

Built to demonstrate real-world AI inference optimization techniques used in production ML systems and AI accelerator pipelines.

---

## 🚀 What It Does

Takes any text input and runs it through **3 versions of the same NLP model** (DistilBERT sentiment classifier), returning:

| Variant | Format | What changes |
|---|---|---|
| **Baseline** | PyTorch `.bin` | Standard HuggingFace model, no optimization |
| **ONNX** | `.onnx` | Exported + graph-optimized by ONNX Runtime |
| **Quantized** | INT8 `.onnx` | Weights compressed from FP32 → INT8 |

Each request returns latency (ms), model size (MB), prediction label, and confidence — side by side.

---

## 🧠 Key Concepts Demonstrated

### 1. ONNX Export
Converts the PyTorch model to ONNX (Open Neural Network Exchange) format. ONNX Runtime applies:
- **Operator fusion** — merges multiple ops into one (e.g. LayerNorm fusion)
- **Memory planning** — reduces allocations during inference
- **Hardware-agnostic optimization** — runs efficiently on any CPU/GPU

### 2. INT8 Dynamic Quantization
Reduces weight precision from 32-bit floats → 8-bit integers:
- **4× smaller** model file
- **2–4× faster** inference on CPU
- **<1% accuracy drop** on most NLP classification tasks
- No need for a calibration dataset (dynamic quantization)

### 3. Why This Matters for AI Accelerators
AI accelerator teams optimize model inference for deployment at scale — on CPUs, GPUs, NPUs, and custom silicon. The techniques here (ONNX, quantization, batching) are the foundation of what systems like NVIDIA TensorRT, Intel OpenVINO, and HCL's accelerator stack do.

---

## 📁 Project Structure

```
model-speed-comparator/
├── app/
│   ├── main.py          # FastAPI routes
│   └── inference.py     # All 3 model variants + benchmarking
├── static/
│   └── index.html       # Dashboard UI
├── models/              # Auto-generated on first run
│   ├── model.onnx
│   └── model_quantized.onnx
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Setup & Run

### Local

```bash
# 1. Clone and install
git clone https://github.com/yourusername/model-speed-comparator
cd model-speed-comparator
pip install -r requirements.txt

# 2. Run
uvicorn app.main:app --reload --port 8000

# 3. Open browser
# http://localhost:8000
```

> **First run:** ONNX export + quantization happens automatically (~2 min one-time setup). Models are cached in `/models`.

### Docker

```bash
docker build -t model-speed-comparator .
docker run -p 8000:8000 model-speed-comparator
```

---

## 📡 API

### `POST /compare`

```json
// Request
{ "text": "This product is absolutely amazing!" }

// Response
{
  "input": "This product is absolutely amazing!",
  "results": {
    "baseline":  { "label": "POSITIVE", "confidence": 0.9998, "latency_ms": 118.4, "model_size_mb": 268.0, "format": "PyTorch (.bin)" },
    "onnx":      { "label": "POSITIVE", "confidence": 0.9997, "latency_ms": 58.2,  "model_size_mb": 268.0, "format": "ONNX (.onnx)" },
    "quantized": { "label": "POSITIVE", "confidence": 0.9995, "latency_ms": 29.1,  "model_size_mb": 68.0,  "format": "Quantized ONNX INT8 (.onnx)" }
  },
  "summary": {
    "fastest": "quantized",
    "smallest": "quantized",
    "speedup_vs_baseline": 4.1
  }
}
```

### `GET /health`
Returns `{ "status": "ok" }`

---

## 📊 Typical Results (CPU, DistilBERT)

| Variant | Latency | Size | vs Baseline |
|---|---|---|---|
| PyTorch Baseline | ~120ms | 268MB | 1× |
| ONNX | ~60ms | 268MB | ~2× faster |
| INT8 Quantized | ~30ms | 68MB | ~4× faster, 4× smaller |

---

## 🛠 Tech Stack

- **FastAPI** — Async REST API
- **HuggingFace Transformers** — DistilBERT model
- **ONNX Runtime** — Optimized inference engine
- **Optimum** — HuggingFace's ONNX export toolkit
- **Docker** — Containerized deployment

---

## 🔮 Possible Extensions

- Add more optimization levels (FP16, INT4)
- Support multiple model architectures (BERT, RoBERTa)
- Add batch inference benchmarking
- Integrate OpenVINO as a 4th backend
- Add latency percentile tracking (p50, p95, p99)

---

## 💬 Interview Explanation (30 seconds)

> *"I took DistilBERT, a popular transformer model, and optimized it for CPU deployment in two steps — first converting it to ONNX format for graph-level optimizations, then applying dynamic INT8 quantization. The quantized model runs 4× faster and is 4× smaller with the same prediction accuracy. I wrapped all three variants in a FastAPI service with a live benchmarking dashboard. This directly mirrors what AI accelerator teams do — optimizing model inference for efficient deployment on target hardware."*
