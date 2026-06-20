from datetime import datetime
import time

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.inference import run_all_models, run_benchmark

app = FastAPI(title="Model Speed Comparator", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

comparison_history = []
MAX_HISTORY_ITEMS = 10

class TextInput(BaseModel):
    text: str


class BatchInput(BaseModel):
    texts: list[str]


session_stats = {
    "total_requests": 0,
    "latency_sum": {
        "baseline": 0.0,
        "onnx": 0.0,
        "quantized": 0.0,
    },
    "fastest_count": {
        "baseline": 0,
        "onnx": 0,
        "quantized": 0,
    },
}


def _build_summary(results: dict) -> dict:
    quantized_latency = max(results["quantized"]["latency_ms"], 0.01)
    return {
        "fastest": min(results, key=lambda x: results[x]["latency_ms"]),
        "smallest": min(results, key=lambda x: results[x]["model_size_mb"]),
        "speedup_vs_baseline": round(
            results["baseline"]["latency_ms"] / quantized_latency, 1
        )
    }


def _model_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=(
            "Model loading or inference failed. Confirm dependencies are installed, "
            "run the server from the project root, and check the models folder. "
            f"Original error: {exc}"
        ),
    )


def _add_history_item(text: str, results: dict, summary: dict) -> None:
    comparison_history.insert(
        0,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input": text,
            "results": results,
            "summary": summary,
        },
    )
    del comparison_history[MAX_HISTORY_ITEMS:]


def _record_stats(results: dict, summary: dict) -> None:
    session_stats["total_requests"] += 1
    session_stats["fastest_count"][summary["fastest"]] += 1
    for model_name, result in results.items():
        session_stats["latency_sum"][model_name] += result["latency_ms"]


def _run_comparison(text: str, include_history: bool = True) -> dict:
    results = run_all_models(text)
    summary = _build_summary(results)
    _record_stats(results, summary)
    if include_history:
        _add_history_item(text, results, summary)
    return {
        "input": text,
        "results": results,
        "summary": summary,
    }


def _validate_batch_texts(texts: list[str]) -> list[str]:
    if not texts:
        raise HTTPException(status_code=400, detail="At least one text is required")
    if len(texts) > 10:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 10 texts")
    cleaned = [text.strip() for text in texts]
    if any(not text for text in cleaned):
        raise HTTPException(status_code=400, detail="Batch texts cannot be empty")
    return cleaned

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/compare")
async def compare_models(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        return _run_comparison(input.text)
    except Exception as exc:
        raise _model_error(exc) from exc


@app.post("/compare/batch")
async def compare_batch(input: BatchInput):
    texts = _validate_batch_texts(input.texts)
    started = time.perf_counter()
    items = []

    try:
        for text in texts:
            items.append(_run_comparison(text))
    except Exception as exc:
        raise _model_error(exc) from exc

    total_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "count": len(items),
        "total_wall_time_ms": total_ms,
        "avg_wall_time_per_text_ms": round(total_ms / len(items), 2),
        "throughput_texts_per_second": round(len(items) / (total_ms / 1000), 2) if total_ms > 0 else 0,
        "items": items,
    }

@app.post("/benchmark")
async def benchmark_models(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        benchmark = run_benchmark(input.text, iterations=20)
    except Exception as exc:
        raise _model_error(exc) from exc

    summary = _build_summary(benchmark["latest_results"])

    return {
        "input": input.text,
        "summary": summary,
        **benchmark,
    }

@app.get("/history")
async def get_history():
    return {"history": comparison_history}

@app.get("/stats")
async def get_stats():
    total = session_stats["total_requests"]
    avg_latency = {}
    for model_name, latency_sum in session_stats["latency_sum"].items():
        avg_latency[model_name] = round(latency_sum / total, 2) if total else 0

    return {
        "total_requests": total,
        "avg_latency": avg_latency,
        "fastest_count": session_stats["fastest_count"],
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
