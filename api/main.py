import torch
import numpy as np
import cv2
import io
import time
import base64
import os
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model, get_device
from src.gradcam import GradCAM, overlay_heatmap

# ─────────────────────────────────────────
# App initialization
# ─────────────────────────────────────────
app = FastAPI(
    title="Manufacturing Defect Detection API",
    description="""
    AI-powered quality inspection system for manufacturing operations.
    
    Analyzes casting product images and classifies them as:
    - **DEFECTIVE** — part should be rejected
    - **NORMAL** — part passes quality inspection
    
    Includes Grad-CAM explainability showing WHERE the model
    detected the defect on the casting surface.
    
    Built with ResNet18 transfer learning — 99.86% accuracy
    on held-out test data.
    """,
    version="1.0.0"
)

# ─────────────────────────────────────────
# Global model loading
# ─────────────────────────────────────────
print("Loading defect detection model...")
DEVICE = get_device()
CLASSES = ['DEFECTIVE', 'NORMAL']
MODEL = None
GRADCAM = None

def load_model():
    """Load model once at startup."""
    global MODEL, GRADCAM
    model = build_model(num_classes=2, freeze_backbone=False)
    model.load_state_dict(
        torch.load('models/best_model.pth', map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()
    
    # Attach Grad-CAM to last conv layer
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)
    
    MODEL = model
    GRADCAM = gradcam
    print("Model loaded successfully!")

# Load on startup
load_model()

# ─────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image_bytes(image_bytes):
    """Convert uploaded image bytes to model input tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    return image, tensor

def heatmap_to_base64(image, heatmap):
    """Convert heatmap overlay to base64 string for JSON response."""
    overlay = overlay_heatmap(image, heatmap)
    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ─────────────────────────────────────────
# Response models
# ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    defective_probability: float
    normal_probability: float
    latency_ms: float
    gradcam_overlay: str  # base64 encoded PNG
    model_version: str
    recommendation: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_accuracy: str

class BatchSummary(BaseModel):
    total_images: int
    defective_count: int
    normal_count: int
    defect_rate: float
    average_confidence: float
    average_latency_ms: float

# ─────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint — API health check."""
    return HealthResponse(
        status="operational",
        model_loaded=MODEL is not None,
        device=str(DEVICE),
        model_accuracy="99.86%"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring systems."""
    return HealthResponse(
        status="operational",
        model_loaded=MODEL is not None,
        device=str(DEVICE),
        model_accuracy="99.86%"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Analyze a casting image for manufacturing defects.
    
    Upload an image of a casting product and receive:
    - Classification (DEFECTIVE or NORMAL)
    - Confidence score
    - Grad-CAM heatmap overlay showing detected defect location
    - Recommendation for quality control action
    
    Supported formats: JPG, JPEG, PNG
    """
    # Validate file type
    if file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Please upload JPG or PNG images."
        )
    
    # Read image
    try:
        image_bytes = await file.read()
        image, tensor = preprocess_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(e)}"
        )
    
    # Run inference with timing
    start_time = time.time()
    
    try:
        heatmap, pred_class, confidence = GRADCAM.generate(tensor)
        gradcam_b64 = heatmap_to_base64(image, heatmap)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Get probabilities
    with torch.no_grad():
        output = MODEL(tensor)
        probs = torch.softmax(output, dim=1)[0]
        defective_prob = probs[0].item()
        normal_prob = probs[1].item()
    
    prediction = CLASSES[pred_class]
    
    # Generate recommendation
    if prediction == 'DEFECTIVE':
        if confidence >= 0.95:
            recommendation = ("REJECT — High confidence defect detected. "
                            "Remove from production line immediately.")
        else:
            recommendation = ("REVIEW — Possible defect detected. "
                            "Flag for manual inspection.")
    else:
        if confidence >= 0.95:
            recommendation = ("PASS — High confidence normal part. "
                            "Clear for next production stage.")
        else:
            recommendation = ("REVIEW — Classified as normal but with "
                            "lower confidence. Consider manual check.")
    
    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        defective_probability=round(defective_prob * 100, 2),
        normal_probability=round(normal_prob * 100, 2),
        latency_ms=round(latency_ms, 2),
        gradcam_overlay=gradcam_b64,
        model_version="ResNet18-v1.0",
        recommendation=recommendation
    )

@app.post("/predict/batch-summary")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Analyze multiple casting images and return summary statistics.
    
    Useful for end-of-shift quality reports or batch inspection.
    Returns defect rate, counts, and average confidence across
    all submitted images.
    """
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 images per batch request."
        )
    
    results = []
    total_latency = 0
    
    for file in files:
        if file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
            continue
        
        try:
            image_bytes = await file.read()
            image, tensor = preprocess_image_bytes(image_bytes)
            
            start = time.time()
            heatmap, pred_class, confidence = GRADCAM.generate(tensor)
            latency = (time.time() - start) * 1000
            
            results.append({
                'prediction': CLASSES[pred_class],
                'confidence': confidence,
                'latency_ms': latency
            })
            total_latency += latency
            
        except Exception:
            continue
    
    if not results:
        raise HTTPException(
            status_code=400,
            detail="No valid images could be processed."
        )
    
    defective = [r for r in results if r['prediction'] == 'DEFECTIVE']
    normal = [r for r in results if r['prediction'] == 'NORMAL']
    
    return BatchSummary(
        total_images=len(results),
        defective_count=len(defective),
        normal_count=len(normal),
        defect_rate=round(len(defective) / len(results) * 100, 2),
        average_confidence=round(
            sum(r['confidence'] for r in results) / len(results) * 100, 2
        ),
        average_latency_ms=round(total_latency / len(results), 2)
    )

@app.get("/model/info")
async def model_info():
    """Return model architecture and performance information."""
    return {
        "model_architecture": "ResNet18 with custom classifier head",
        "training_approach": "Two-phase transfer learning",
        "phase_1": "Frozen backbone, classifier head only (5 epochs)",
        "phase_2": "Full fine-tuning, all layers (5 epochs)",
        "dataset": "Real-Life Industrial Dataset of Casting Products",
        "training_samples": 6633,
        "test_samples": 715,
        "performance": {
            "accuracy": "99.86%",
            "precision": "99.86%",
            "recall": "99.86%",
            "f1_score": "99.86%",
            "defect_recall": "99.78%"
        },
        "classes": {
            "0": "DEFECTIVE — casting has surface defect",
            "1": "NORMAL — casting passes quality inspection"
        },
        "explainability": "Grad-CAM heatmaps on final conv layer",
        "deployment": "FastAPI REST API in Docker container"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)