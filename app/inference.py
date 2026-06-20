"""
inference.py — Core logic for all 3 model variants.

Variants:
  1. Baseline   — Standard HuggingFace PyTorch model
  2. ONNX       — Exported to ONNX format (faster runtime)
  3. Quantized  — INT8 quantized ONNX (fastest + smallest)

Each variant returns: label, confidence, latency_ms, model_size_mb
"""

import math
import time
import os
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")
ONNX_PATH = MODEL_DIR / "model.onnx"
QUANTIZED_PATH = MODEL_DIR / "model_quantized.onnx"

_baseline_pipeline = None
_onnx_session = None
_quantized_session = None
_tokenizer = None


def _get_file_size_mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 1)
    except FileNotFoundError:
        return 0.0


def _load_baseline():
    global _baseline_pipeline
    if _baseline_pipeline is None:
        from transformers import pipeline
        print("[INFO] Loading baseline PyTorch model...")
        _baseline_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
    return _baseline_pipeline


def _load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _tokenizer


def _export_to_onnx():
    """Export model to ONNX using optimum library."""
    if not ONNX_PATH.exists():
        print("[INFO] Exporting model to ONNX (one-time setup)...")
        MODEL_DIR.mkdir(exist_ok=True)
        from optimum.onnxruntime import ORTModelForSequenceClassification
        model = ORTModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english",
            export=True
        )
        model.save_pretrained(str(MODEL_DIR))
        # Find and rename if needed
        if not ONNX_PATH.exists():
            import shutil
            for f in MODEL_DIR.glob("*.onnx"):
                shutil.copy(str(f), str(ONNX_PATH))
                break
        print(f"[INFO] ONNX model saved to {ONNX_PATH}")


def _quantize_onnx():
    """Quantize ONNX model to INT8 if not already done."""
    if not QUANTIZED_PATH.exists():
        _export_to_onnx()
        print("[INFO] Quantizing ONNX model to INT8...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            str(ONNX_PATH),
            str(QUANTIZED_PATH),
            weight_type=QuantType.QInt8
        )
        print(f"[INFO] Quantized model saved to {QUANTIZED_PATH}")


def _load_onnx_session(path: str):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=opts)


def _run_onnx_inference(session, text: str) -> dict:
    tokenizer = _load_tokenizer()
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
    feed = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
    outputs = session.run(None, feed)
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    label_idx = int(np.argmax(probs))
    labels = ["NEGATIVE", "POSITIVE"]
    return {
        "label": labels[label_idx],
        "confidence": round(float(probs[label_idx]), 4)
    }


def run_baseline(text: str) -> dict:
    pipeline = _load_baseline()
    start = time.perf_counter()
    result = pipeline(text)[0]
    latency = (time.perf_counter() - start) * 1000
    return {
        "label": result["label"],
        "confidence": round(result["score"], 4),
        "latency_ms": round(latency, 2),
        "model_size_mb": 268.0,
        "format": "PyTorch (.bin)"
    }


def run_onnx(text: str) -> dict:
    global _onnx_session
    _export_to_onnx()
    if _onnx_session is None:
        _onnx_session = _load_onnx_session(ONNX_PATH)
    start = time.perf_counter()
    prediction = _run_onnx_inference(_onnx_session, text)
    latency = (time.perf_counter() - start) * 1000
    return {
        **prediction,
        "latency_ms": round(latency, 2),
        "model_size_mb": _get_file_size_mb(ONNX_PATH) or 268.0,
        "format": "ONNX (.onnx)"
    }


def run_quantized(text: str) -> dict:
    global _quantized_session
    _quantize_onnx()
    if _quantized_session is None:
        _quantized_session = _load_onnx_session(QUANTIZED_PATH)
    start = time.perf_counter()
    prediction = _run_onnx_inference(_quantized_session, text)
    latency = (time.perf_counter() - start) * 1000
    return {
        **prediction,
        "latency_ms": round(latency, 2),
        "model_size_mb": _get_file_size_mb(QUANTIZED_PATH) or 68.0,
        "format": "Quantized ONNX INT8 (.onnx)"
    }


def run_all_models(text: str) -> dict:
    return {
        "baseline": run_baseline(text),
        "onnx": run_onnx(text),
        "quantized": run_quantized(text),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil((percentile / 100) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def run_benchmark(text: str, iterations: int = 20) -> dict:
    latencies = {
        "baseline": [],
        "onnx": [],
        "quantized": [],
    }
    latest_results = None

    for _ in range(iterations):
        latest_results = run_all_models(text)
        for model_name, result in latest_results.items():
            latencies[model_name].append(result["latency_ms"])

    stats = {}
    for model_name, model_latencies in latencies.items():
        latest = latest_results[model_name] if latest_results else {}
        stats[model_name] = {
            "avg_latency_ms": round(sum(model_latencies) / len(model_latencies), 2),
            "min_latency_ms": round(min(model_latencies), 2),
            "max_latency_ms": round(max(model_latencies), 2),
            "p95_latency_ms": round(_percentile(model_latencies, 95), 2),
            "model_size_mb": latest.get("model_size_mb", 0.0),
            "format": latest.get("format", ""),
        }

    return {
        "iterations": iterations,
        "latest_results": latest_results,
        "results": stats,
    }
