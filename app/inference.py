import time, os, gc, numpy as np
from pathlib import Path

MODEL_DIR = Path("models")
ONNX_PATH = MODEL_DIR / "model.onnx"
QUANTIZED_PATH = MODEL_DIR / "model_quantized.onnx"
MODEL_NAME = "prajjwal1/bert-tiny"
_tokenizer = None

def _get_file_size_mb(path) -> float:
    try: return round(os.path.getsize(path) / (1024*1024), 1)
    except: return 0.0

def _load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer

def _export_to_onnx():
    if not ONNX_PATH.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        from optimum.onnxruntime import ORTModelForSequenceClassification
        model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAME, export=True)
        model.save_pretrained(str(MODEL_DIR))
        if not ONNX_PATH.exists():
            import shutil
            for f in MODEL_DIR.glob("*.onnx"):
                shutil.copy(str(f), str(ONNX_PATH)); break
        del model; gc.collect()

def _quantize_onnx():
    if not QUANTIZED_PATH.exists():
        _export_to_onnx()
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(ONNX_PATH), str(QUANTIZED_PATH), weight_type=QuantType.QInt8)

def _onnx_infer(session, text):
    tok = _load_tokenizer()
    inputs = tok(text, return_tensors="np", truncation=True, max_length=128, padding=True)
    feed = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
    out = session.run(None, feed)[0][0]
    exp = np.exp(out - np.max(out)); probs = exp / exp.sum()
    idx = int(np.argmax(probs))
    return {"label": ["NEGATIVE","POSITIVE"][idx], "confidence": round(float(probs[idx]),4)}

def run_baseline(text):
    from transformers import pipeline
    pipe = pipeline("text-classification", model=MODEL_NAME, device=-1)
    start = time.perf_counter()
    result = pipe(text)[0]
    latency = (time.perf_counter()-start)*1000
    del pipe; gc.collect()
    label = "POSITIVE" if result["label"] == "LABEL_1" else "NEGATIVE"
    return {"label": label, "confidence": round(result["score"],4),
            "latency_ms": round(latency,2), "model_size_mb": 17.0, "format": "PyTorch (.bin)"}

def run_onnx(text):
    import onnxruntime as ort
    _export_to_onnx()
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts)
    start = time.perf_counter()
    pred = _onnx_infer(session, text)
    latency = (time.perf_counter()-start)*1000
    del session; gc.collect()
    return {**pred, "latency_ms": round(latency,2),
            "model_size_mb": _get_file_size_mb(ONNX_PATH) or 17.0, "format": "ONNX (.onnx)"}

def run_quantized(text):
    import onnxruntime as ort
    _quantize_onnx()
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(QUANTIZED_PATH), sess_options=opts)
    start = time.perf_counter()
    pred = _onnx_infer(session, text)
    latency = (time.perf_counter()-start)*1000
    del session; gc.collect()
    return {**pred, "latency_ms": round(latency,2),
            "model_size_mb": _get_file_size_mb(QUANTIZED_PATH) or 4.5, "format": "Quantized ONNX INT8 (.onnx)"}

def run_all_models(text):
    b = run_baseline(text); gc.collect()
    o = run_onnx(text); gc.collect()
    q = run_quantized(text); gc.collect()
    return {"baseline": b, "onnx": o, "quantized": q}

import math

def _percentile(values, p):
    if not values: return 0.0
    s = sorted(values)
    i = max(0, min(math.ceil((p/100)*len(s))-1, len(s)-1))
    return s[i]

def run_benchmark(text: str, iterations: int = 20) -> dict:
    latencies = {"baseline": [], "onnx": [], "quantized": []}
    latest = None
    for _ in range(iterations):
        latest = run_all_models(text)
        for k, v in latest.items():
            latencies[k].append(v["latency_ms"])
    stats = {}
    for k, lats in latencies.items():
        stats[k] = {
            "avg_latency_ms": round(sum(lats)/len(lats), 2),
            "min_latency_ms": round(min(lats), 2),
            "max_latency_ms": round(max(lats), 2),
            "p95_latency_ms": round(_percentile(lats, 95), 2),
            "model_size_mb": latest[k]["model_size_mb"],
            "format": latest[k]["format"],
        }
    return {"iterations": iterations, "latest_results": latest, "results": stats}
