"""
OvoScan AI Inference Service - FastAPI backend.

FastAPI with:
- YOLOv8 classification at /predict
- RAG safety report at /analyze-report
- Prometheus metrics at /metrics

Usage:
    uvicorn src.app:app --host 0.0.0.0 --port 8001 --reload
"""

import io
import os
import time
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from src.auth import auth_enabled, require_api_key
from src.uploads import enforce_content_length, read_upload

# Import monitoring utilities
from src.utils.monitoring import (
    get_monitoring_summary,
    record_inference_error,
    record_prediction,
    set_model_loaded,
)

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Try to import RAG agent
try:
    from src.agent.rag import HatcheryAgent
    AGENT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    AGENT_AVAILABLE = False
    HatcheryAgent = None

# Global variables
model = None
agent = None
torch_device = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy AI models only once when the server starts.
    Includes intelligent hardware detection (CPU vs GPU).
    """
    global model, agent, torch_device

    # 1. Hardware Check
    if torch.cuda.is_available():
        torch_device = "cuda"
        print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print("System running in ACCELERATED mode")
    else:
        torch_device = "cpu"
        print("GPU not detected. System running in CPU mode")

    # 2. Load YOLO Model
    model_path = os.getenv("MODEL_PATH", "serving/model.pt")
    fallback_path = os.getenv("FALLBACK_MODEL", "yolov8n-cls.pt")

    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"Model loaded from {model_path}")
        set_model_loaded(True)
    elif os.path.exists(fallback_path):
        model = YOLO(fallback_path)
        print(f"Using fallback model at {fallback_path}")
        set_model_loaded(True)
    else:
        print("No model found - classifier unavailable")
        model = None
        set_model_loaded(False)

    # 3. Load RAG Agent
    kb_path = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base/hatchery_manual.txt")
    if AGENT_AVAILABLE and os.path.exists(kb_path):
        try:
            agent = HatcheryAgent(kb_path)
            agent.ingest_knowledge()
            print("Knowledge base loaded")
        except Exception as e:
            print(f"Knowledge base loading failed: {e}")
            agent = None
    else:
        print("RAG agent unavailable (knowledge base or module missing)")

    yield

    # Cleanup
    print("Shutting down AI Services...")


def _detect_image(file_bytes: bytes) -> dict:
    """Run YOLO detection on image bytes."""
    if model is None:
        raise RuntimeError("Model not loaded")

    image = Image.open(io.BytesIO(file_bytes))

    # Run inference with timing
    start_time = time.time()
    results = model.predict(image, conf=0.5)
    latency = time.time() - start_time

    result = results[0]

    top_class_id = int(result.probs.top1)
    top_class_name = result.names[top_class_id]
    confidence = float(result.probs.top1conf)

    # Record metrics
    record_prediction(top_class_name, confidence, latency)

    return {
        "prediction": top_class_name,
        "confidence": round(confidence, 4),
    }


def _generate_report(prediction: str, confidence: float) -> str:
    """Generate technical report using RAG agent or template."""
    if agent is not None:
        try:
            return agent.analyze_defect(prediction)
        except Exception as e:
            print(f"RAG analysis failed: {e}")

    # Template-based fallback
    if prediction.lower() == "fertile":
        return "Quality Standard Met. Ready for incubation."
    return f"Defect detected: {prediction} (confidence: {confidence:.2f}). Recommend rejection."


# Initialize FastAPI
app = FastAPI(
    title="OvoScan AI Inference Service",
    description="Industrial Defect Detection API with RAG Reporting",
    version="2.1.0",
    lifespan=lifespan,
)


@app.get("/")
def health_check():
    """Heartbeat endpoint."""
    return {
        "status": "running",
        "service": "ovoscan-ai-v2",
        "model_loaded": model is not None,
        "rag_available": agent is not None,
        "device": torch_device,
        "auth_enabled": auth_enabled(),
    }


@app.post("/predict", dependencies=[Depends(require_api_key), Depends(enforce_content_length)])
async def predict(file: UploadFile = File(...)):
    """Accepts an image, runs inference, and returns prediction."""
    try:
        contents = await read_upload(file)
        result = _detect_image(contents)

        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "status": "success",
        }

    except RuntimeError as e:
        record_inference_error()
        return {"status": "error", "message": f"Model not available: {e}"}
    except Exception as e:
        record_inference_error()
        return {"status": "error", "message": str(e)}


@app.post("/analyze-report", dependencies=[Depends(require_api_key), Depends(enforce_content_length)])
async def analyze_report(file: UploadFile = File(...)):
    """Full safety report: detection + RAG analysis."""
    try:
        contents = await read_upload(file)
        result = _detect_image(contents)

        report = _generate_report(result["prediction"], result["confidence"])

        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "technical_report": report,
            "status": "success",
        }

    except RuntimeError as e:
        return {"status": "error", "message": f"Service unavailable: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring", dependencies=[Depends(require_api_key)])
def monitoring_summary():
    """Get monitoring summary for debugging and dashboards."""
    return get_monitoring_summary()


@app.get("/drift", dependencies=[Depends(require_api_key)])
def drift_summary():
    """Drift state under the name the other three projects use.

    This project originally exposed the same payload only at /monitoring, so
    the README, the retraining DAG and the shared documentation all disagreed
    about the endpoint name. /monitoring is kept because the DAG and the
    dashboards already call it.
    """
    return get_monitoring_summary()


@app.post("/admin/reload-model", dependencies=[Depends(require_api_key)])
def reload_model():
    """Re-read the classifier weights from disk without restarting the API.

    The retraining DAG writes new weights, then calls this so the fresh model is
    served immediately. Disabled unless ENABLE_MODEL_RELOAD=true: swapping model
    weights is a privileged operation and should not be reachable by default.
    """
    global model

    if os.getenv("ENABLE_MODEL_RELOAD", "false").lower() not in {"1", "true", "yes"}:
        raise HTTPException(
            status_code=403,
            detail="Model reload is disabled. Set ENABLE_MODEL_RELOAD=true to enable it.",
        )

    model_path = os.getenv("MODEL_PATH", "serving/model.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"No model at {model_path}")

    try:
        model = YOLO(model_path)
        set_model_loaded(True)
    except Exception as e:  # noqa: BLE001 - surface the reason to the caller
        set_model_loaded(False)
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e

    return {"status": "reloaded", "model_path": model_path}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
