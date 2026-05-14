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
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "./llama-drift-merged"
MAX_SEQ_LEN    = 320
LABEL_NAMES    = ["no_drift", "drift"]
CACHE_MAX_SIZE = 512
BATCH_MAX_SIZE = 32

# ── Model state ───────────────────────────────────────────────────────────────
_model     = None
_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    try:
        log.info("Loading tokenizer from %s ...", MODEL_PATH)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        log.info("Loading model...")
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=2,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _model.eval()
        log.info("Model ready on device: %s", next(_model.parameters()).device)
    except OSError as e:
        log.critical("Model path not found: %s — %s", MODEL_PATH, e)
        raise RuntimeError(f"Cannot load model from '{MODEL_PATH}': {e}") from e
    except Exception as e:
        log.critical("Unexpected error during model load: %s", e)
        raise

    yield

    log.info("Shutting down — releasing model.")
    del _model, _tokenizer


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


# ── Global exception handlers ─────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return a clean 422 with field-level detail instead of FastAPI's default."""
    errors = [
        {"field": ".".join(str(l) for l in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    log.warning("Validation error on %s: %s", request.url.path, errors)
    return JSONResponse(status_code=422, content={"detail": "Validation failed.", "errors": errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception on %s:\n%s", request.url.path, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=256, example="budget discussion")
    transcript: str = Field(..., min_length=10, max_length=4096, example="A: We are over budget. B: By how much?")

    @field_validator("topic")
    @classmethod
    def topic_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must not be blank.")
        return v.strip()

    @field_validator("transcript")
    @classmethod
    def transcript_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("transcript must not be blank.")
        return v.strip()


class PredictResponse(BaseModel):
    label:         str
    drift:         bool
    confidence:    float
    no_drift_prob: float
    drift_prob:    float
    cached:        bool = False
    latency_ms:    float


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=BATCH_MAX_SIZE)

    @field_validator("items")
    @classmethod
    def items_not_empty(cls, v):
        if not v:
            raise ValueError("items list must not be empty.")
        return v


class BatchPredictResponse(BaseModel):
    results:    list[PredictResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    status:     str
    model_path: str
    device:     str
    cuda:       bool


class ErrorResponse(BaseModel):
    detail: str


# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}

def _cache_key(topic: str, transcript: str) -> str:
    return hashlib.sha256(f"{topic}||{transcript}".encode()).hexdigest()

def _cache_get(key: str) -> Optional[dict]:
    return _cache.get(key)

def _cache_set(key: str, value: dict):
    if len(_cache) >= CACHE_MAX_SIZE:
        oldest = next(iter(_cache))
        del _cache[oldest]
    _cache[key] = value


# ── Core inference ────────────────────────────────────────────────────────────
def _run_inference(topic: str, transcript: str) -> dict:
    try:
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
    except torch.cuda.OutOfMemoryError as e:
        log.error("CUDA OOM during inference: %s", e)
        raise HTTPException(status_code=507, detail="GPU out of memory. Try a shorter transcript or use batch endpoint.")
    except Exception as e:
        log.error("Inference failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


def _run_batch_inference(items: list[tuple[str, str]]) -> list[dict]:
    try:
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
    except torch.cuda.OutOfMemoryError as e:
        log.error("CUDA OOM during batch inference: %s", e)
        raise HTTPException(
            status_code=507,
            detail="GPU out of memory. Reduce batch size or transcript lengths.",
        )
    except Exception as e:
        log.error("Batch inference failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch inference error: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["System"],
)
def health():
    if _model is None or _tokenizer is None:
        log.warning("Health check called but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        device = str(next(_model.parameters()).device)
    except StopIteration:
        raise HTTPException(status_code=503, detail="Model has no parameters — may be corrupted.")
    return {"status": "ok", "model_path": MODEL_PATH, "device": device, "cuda": torch.cuda.is_available()}


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        507: {"model": ErrorResponse, "description": "GPU out of memory"},
        500: {"model": ErrorResponse, "description": "Inference error"},
    },
    tags=["Inference"],
)
def predict(req: PredictRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0  = time.perf_counter()
    key = _cache_key(req.topic, req.transcript)

    hit = _cache_get(key)
    if hit:
        log.info("Cache hit for predict request.")
        return PredictResponse(**hit, cached=True, latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    result = _run_inference(req.topic, req.transcript)
    _cache_set(key, result)
    log.info("Predict: drift=%s conf=%.4f latency=%.1fms", result["drift"], result["confidence"], (time.perf_counter() - t0) * 1000)
    return PredictResponse(**result, cached=False, latency_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Empty batch"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        507: {"model": ErrorResponse, "description": "GPU out of memory"},
        500: {"model": ErrorResponse, "description": "Inference error"},
    },
    tags=["Inference"],
)
def predict_batch(req: BatchPredictRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0          = time.perf_counter()
    results     = [None] * len(req.items)
    miss_idx    = []
    miss_inputs = []

    for i, item in enumerate(req.items):
        key = _cache_key(item.topic, item.transcript)
        hit = _cache_get(key)
        if hit:
            results[i] = PredictResponse(**hit, cached=True, latency_ms=0)
        else:
            miss_idx.append(i)
            miss_inputs.append((item.topic, item.transcript))

    if miss_inputs:
        inferred = _run_batch_inference(miss_inputs)
        for idx, result in zip(miss_idx, inferred):
            key = _cache_key(req.items[idx].topic, req.items[idx].transcript)
            _cache_set(key, result)
            results[idx] = PredictResponse(**result, cached=False, latency_ms=0)

    # Catch any None slots that shouldn't exist
    if any(r is None for r in results):
        log.error("Some batch results are None — partial inference failure.")
        raise HTTPException(status_code=500, detail="Partial batch inference failure.")

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info("Batch predict: %d items (%d cache hits) latency=%.1fms",
             len(req.items), len(req.items) - len(miss_idx), total_ms)
    return BatchPredictResponse(results=results, latency_ms=total_ms)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)