import io
import os
import torch  # <--- NEW IMPORT
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO
import uvicorn
from contextlib import asynccontextmanager

# Import our new RAG Agent
from src.agent.rag import HatcheryAgent

# Global Variables
model = None
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy AI models only once when the server starts.
    Includes intelligent hardware detection (CPU vs GPU).
    """
    global model, agent
    
    # ---------------------------------------------------------
    # 1. HARDWARE CHECK (The Fallback Logic)
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f" GPU DETECTED: {gpu_name}")
        print(" System is running in ACCELERATED mode.")
    else:
        device = "cpu"
        print(" NO GPU DETECTED.")
        print(" System is falling back to CPU mode.")
    # ---------------------------------------------------------

    # 2. Load YOLO Model
    print(f" Loading YOLOv8 Model on {device.upper()}...")
    model_path = "serving/model.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        print(" Training weights not found, using default...")
        model = YOLO("yolov8n-cls.pt")
        
    # 3. Load RAG Agent
    print(" Initializing Hatchery Knowledge Agent...")
    kb_path = "data/knowledge_base/manual.txt"
    if os.path.exists(kb_path):
        agent = HatcheryAgent(kb_path)
        agent.ingest_knowledge()
    else:
        print(" Knowledge base not found. RAG functionality disabled.")
    
    yield
    
    # Cleanup
    print(" Shutting down AI Services...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="OvoScan AI Inference Service",
    description="Industrial Defect Detection API with RAG Reporting",
    version="2.1.0",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    return {"status": "running", "service": "ovoscan-ai-v2"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image, runs inference, and generates a technical report if needed.
    """
    try:
        # 1. Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 2. Inference
        # YOLO automatically uses the device detected at load time, 
        # but explicit checks are handled internally by Ultralytics.
        results = model.predict(image, conf=0.5)
        result = results[0]
        
        top_class_id = result.probs.top1
        top_class_name = result.names[top_class_id]
        confidence = float(result.probs.top1conf)
        
        # 3. RAG Analysis (The "Smart" Part)
        report = "N/A"
        if top_class_name == "fertile":
            report = " Quality Standard Met. Ready for incubation."
        elif agent:
            # Ask the LLM why this is bad
            print(f" Defect Detected ({top_class_name}). Consulting Manual...")
            report = agent.analyze_defect(top_class_name)
        
        return {
            "filename": file.filename,
            "prediction": top_class_name,
            "confidence": round(confidence, 4),
            "technical_report": report,
            "status": "success"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Run on port 8001 to avoid conflicts
    uvicorn.run(app, host="0.0.0.0", port=8001)