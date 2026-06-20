"""
inference.py — Core logic for all 3 model variants.
Memory-optimized for 512MB RAM (Render free tier).

Strategy: Load one model at a time, unload before loading next.
"""

import time
import os
import gc
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")
ONNX_PATH = MODEL_DIR / "model.onnx"
QUANTIZED_PATH = MODEL_DIR / "model_quantized.onnx"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_tokenizer = None


def _get_file_size_mb(path) -> float:
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 1)
    except FileNotFoundError:
        return 0.0


def _load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def _export_to_onnx():
    if not ONNX_PATH.exists():
        print("[INFO] Exporting to ONNX...")
        MODEL_DIR.mkdir(exist_ok=True)
        from optimum.onnxruntime import ORTModelForSequenceClassification
        model = ORTModelForSequenceClassification.from_pretrained(
            MODEL_NAME, export=True
        )
        model.save_pretrained(str(MODEL_DIR))
        if not ONNX_PATH.exists():
            import shutil
            for f in MODEL_DIR.glob("*.onnx"):
                shutil.copy(str(f), str(ONNX_PATH))
                break
        del model
        gc.collect()


def _quantize_onnx():
    if not QUANTIZED_PATH.exists():
        _export_to_onnx()
        print("[INFO] Quantizing to INT8...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(ONNX_PATH), str(QUANTIZED_PATH), weight_type=QuantType.QInt8)


def _run_onnx_inference(session, text: str) -> dict:
    tokenizer = _load_tokenizer()
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
    feed = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
    outputs = session.run(None, feed)
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    label_idx = int(np.argmax(probs))
    return {
        "label": ["NEGATIVE", "POSITIVE"][label_idx],
        "confidence": round(float(probs[label_idx]), 4)
    }


def run_baseline(text: str) -> dict:
    from transformers import pipeline
    print("[INFO] Loading baseline...")
    pipe = pipeline("sentiment-analysis", model=MODEL_NAME, device=-1)
    start = time.perf_counter()
    result = pipe(text)[0]
    latency = (time.perf_counter() - start) * 1000
    del pipe
    gc.collect()
    return {
        "label": result["label"],
        "confidence": round(result["score"], 4),
        "latency_ms": round(latency, 2),
        "model_size_mb": 268.0,
        "format": "PyTorch (.bin)"
    }


def run_onnx(text: str) -> dict:
    import onnxruntime as ort
    _export_to_onnx()
    print("[INFO] Loading ONNX...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts)
    start = time.perf_counter()
    prediction = _run_onnx_inference(session, text)
    latency = (time.perf_counter() - start) * 1000
    del session
    gc.collect()
    return {
        **prediction,
        "latency_ms": round(latency, 2),
        "model_size_mb": _get_file_size_mb(ONNX_PATH) or 255.0,
        "format": "ONNX (.onnx)"
    }


def run_quantized(text: str) -> dict:
    import onnxruntime as ort
    _quantize_onnx()
    print("[INFO] Loading quantized...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(QUANTIZED_PATH), sess_options=opts)
    start = time.perf_counter()
    prediction = _run_onnx_inference(session, text)
    latency = (time.perf_counter() - start) * 1000
    del session
    gc.collect()
    return {
        **prediction,
        "latency_ms": round(latency, 2),
        "model_size_mb": _get_file_size_mb(QUANTIZED_PATH) or 64.0,
        "format": "Quantized ONNX INT8 (.onnx)"
    }


def run_all_models(text: str) -> dict:
    baseline = run_baseline(text)
    gc.collect()
    onnx = run_onnx(text)
    gc.collect()
    quantized = run_quantized(text)
    gc.collect()
    return {
        "baseline": baseline,
        "onnx": onnx,
        "quantized": quantized,
    }
