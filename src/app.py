import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="OvoScan AI Inference Service",
    description="Industrial Defect Detection API using YOLOv8",
    version="1.0.0"
)

# Load Model (Global State)
# In production, we load this once at startup
MODEL_PATH = "serving/model.pt"
print(f" Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

@app.get("/")
def health_check():
    """Basic health check endpoint."""
    return {"status": "running", "service": "ovoscan-ai"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, runs inference, and returns the result.
    """
    try:
        # 1. Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 2. Inference
        # conf=0.5 means we only trust predictions with >50% confidence
        results = model.predict(image, conf=0.5)
        
        # 3. Parse Results
        # YOLO returns a list of results (one per image)
        result = results[0]
        
        # Get top class
        top_class_id = result.probs.top1
        top_class_name = result.names[top_class_id]
        confidence = float(result.probs.top1conf)
        
        return {
            "filename": file.filename,
            "prediction": top_class_name,
            "confidence": round(confidence, 4),
            "status": "success"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Run the server locally
    uvicorn.run(app, host="0.0.0.0", port=8000)