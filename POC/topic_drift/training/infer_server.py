"""
FastAPI inference server for the topic drift detection model.

Usage:
    python infer_server.py

Endpoints:
    POST /predict        Single inference
    POST /predict/batch  Batch inference
    GET  /health         Health check
    GET  /docs           Auto-generated Swagger UI
"""

import hashlib
import json
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "./llama-drift-merged"
MAX_SEQ_LEN = 320
LABEL_NAMES = ["no_drift", "drift"]

# Cache: max unique (topic, text) pairs to keep in memory
CACHE_MAX_SIZE = 512

# ── Model state (loaded once at startup) ─────────────────────────────────────
_model     = None
_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _model, _tokenizer
    print("Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    print("Loading model...")
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    _model.eval()
    print(f"Model loaded on device: {next(_model.parameters()).device}")
    yield
    # Cleanup
    del _model, _tokenizer
    print("Model unloaded.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Topic Drift Detection API",
    description="Fine-tuned LLaMA 3.2 model for detecting topic drift in meeting transcripts.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    topic: str = Field(..., example="budget discussion", description="The current meeting agenda item or topic.")
    transcript: str = Field(..., example="A: We are over budget. B: By how much?", description="The meeting transcript window in 'SPEAKER: utterance' format.")


class PredictResponse(BaseModel):
    label:         str
    drift:         bool
    confidence:    float
    no_drift_prob: float
    drift_prob:    float
    cached:        bool = False
    latency_ms:    float


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., max_length=32, description="Up to 32 items per batch request.")


class BatchPredictResponse(BaseModel):
    results:    list[PredictResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    status:     str
    model_path: str
    device:     str
    cuda:       bool


# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}

def _cache_key(topic: str, transcript: str) -> str:
    return hashlib.sha256(f"{topic}||{transcript}".encode()).hexdigest()

def _cache_get(key: str) -> Optional[dict]:
    return _cache.get(key)

def _cache_set(key: str, value: dict):
    if len(_cache) >= CACHE_MAX_SIZE:
        # Evict oldest entry (FIFO)
        oldest = next(iter(_cache))
        del _cache[oldest]
    _cache[key] = value


# ── Core inference ────────────────────────────────────────────────────────────
def _run_inference(topic: str, transcript: str) -> dict:
    combined = f"Topic: {topic}\nTranscript: {transcript}"
    inputs = _tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
    )
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze()
    pred  = logits.argmax(-1).item()

    return {
        "label":         LABEL_NAMES[pred],
        "drift":         pred == 1,
        "confidence":    round(probs[pred].item(), 4),
        "no_drift_prob": round(probs[0].item(), 4),
        "drift_prob":    round(probs[1].item(), 4),
    }


def _run_batch_inference(items: list[tuple[str, str]]) -> list[dict]:
    """
    Tokenise all items together and run a single forward pass.
    More efficient than looping _run_inference for each item.
    """
    texts = [f"Topic: {t}\nTranscript: {tr}" for t, tr in items]
    inputs = _tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
    )
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs_all = torch.softmax(logits, dim=-1)
    preds     = logits.argmax(-1).tolist()

    results = []
    for i, pred in enumerate(preds):
        probs = probs_all[i]
        results.append({
            "label":         LABEL_NAMES[pred],
            "drift":         pred == 1,
            "confidence":    round(probs[pred].item(), 4),
            "no_drift_prob": round(probs[0].item(), 4),
            "drift_prob":    round(probs[1].item(), 4),
        })
    return results


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    device = str(next(_model.parameters()).device)
    return {
        "status":     "ok",
        "model_path": MODEL_PATH,
        "device":     device,
        "cuda":       torch.cuda.is_available(),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0  = time.perf_counter()
    key = _cache_key(req.topic, req.transcript)

    cached = _cache_get(key)
    if cached:
        return PredictResponse(
            **cached,
            cached=True,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    result = _run_inference(req.topic, req.transcript)
    _cache_set(key, result)

    return PredictResponse(
        **result,
        cached=False,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided.")

    t0 = time.perf_counter()

    # Split into cache hits and misses
    results     = [None] * len(req.items)
    miss_idx    = []
    miss_inputs = []

    for i, item in enumerate(req.items):
        key    = _cache_key(item.topic, item.transcript)
        cached = _cache_get(key)
        if cached:
            results[i] = PredictResponse(**cached, cached=True, latency_ms=0)
        else:
            miss_idx.append(i)
            miss_inputs.append((item.topic, item.transcript))

    # Single forward pass for all cache misses
    if miss_inputs:
        inferred = _run_batch_inference(miss_inputs)
        for idx, result in zip(miss_idx, inferred):
            item = req.items[idx]
            key  = _cache_key(item.topic, item.transcript)
            _cache_set(key, result)
            results[idx] = PredictResponse(**result, cached=False, latency_ms=0)

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    return BatchPredictResponse(results=results, latency_ms=total_ms)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)