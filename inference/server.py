
Minimal FastAPI inference server that:
- loads a ModelWrapper on startup
- exposes /health and /predict endpoints
- integrates with SafetyChecker, audit logging, and Prometheus metrics
"""
import logging
import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ..safety_checker import SafetyChecker
from .model_loader import ModelWrapper
from .schemas import InferenceRequest, InferenceResponse, Prediction
from ..metrics.metrics import (
    start_metrics_server,
    INFERENCE_LATENCY_HISTOGRAM,
    INFERENCE_REQUESTS_TOTAL,
    INFERENCE_PREDICTIONS_TOTAL,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="Aegis Inference Service")

# Global singletons (for simplicity). For more advanced setups use dependency injection.
MODEL_PATH = None  # optionally set via env or args
MODEL_VERSION = "dummy-v0"

model_wrapper = ModelWrapper(model_path=MODEL_PATH, model_version=MODEL_VERSION)
safety_checker = SafetyChecker(model_version="safety-heuristic-v0")


@app.on_event("startup")
def startup_event():
    logger.info("Starting Aegis inference service...")
    # start prometheus metrics endpoint (non-blocking)
    try:
        start_metrics_server(port=8000)
        logger.info("Prometheus metrics server started on :8000")
    except Exception as e:
        logger.exception("Failed to start metrics server: %s", e)

    # Load model
    try:
        model_wrapper.load()
        logger.info("Loaded model (version=%s)", model_wrapper.model_version)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_wrapper.model_version}


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    t0 = time.time()
    INFERENCE_REQUESTS_TOTAL.labels(model_version=model_wrapper.model_version).inc()

    texts: List[str] = [item.text for item in req.items]

    # Run safety check on first (or all) items — choose policy: here we check aggregated
    flagged_any = False
    safety_reason = None
    for t in texts:
        if safety_checker.is_unsafe(t):
            flagged_any = True
            safety_reason = "safety_policy"
            break

    # If flagged, optionally deny inference (policy decision). Here we'll return flagged=True and still include no predictions.
    if flagged_any:
        INFERENCE_LATENCY_HISTOGRAM.observe(time.time() - t0)
        # You can choose to block inference completely (raise 403) or return flagged with empty predictions.
        # Here we return 403 to illustrate blocking behavior:
        raise HTTPException(status_code=403, detail="Input flagged by safety policy")

    # Otherwise run model predictions
    try:
        raw_outputs = model_wrapper.predict(texts)
    except Exception as e:
        logger.exception("Model prediction failure: %s", e)
        raise HTTPException(status_code=500, detail="Model inference failed") from e

    predictions = []
    for item, out in zip(req.items, raw_outputs):
        predictions.append(Prediction(id=item.id, output=out))

    INFERENCE_PREDICTIONS_TOTAL.labels(model_version=model_wrapper.model_version).inc(len(predictions))
    INFERENCE_LATENCY_HISTOGRAM.observe(time.time() - t0)

    resp = InferenceResponse(
        flagged=False,
        safety_reason=None,
        model_version=model_wrapper.model_version,
        predictions=predictions,
    )
    return JSONResponse(status_code=200, content=resp.dict())
