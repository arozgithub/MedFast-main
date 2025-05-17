from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
import base64
from PIL import Image
import io
import numpy as np

# Import Vercel configuration if in Vercel environment
if os.environ.get("VERCEL_REGION"):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import vercel
    except ImportError:
        pass

# Initialize FastAPI app
app = FastAPI()

# Simple model for status reporting
class StatusResponse(BaseModel):
    status: str
    message: str

# For handling AI diagnosis requests
class DiagnosisRequest(BaseModel):
    conversation_history: str

# Simple root endpoint
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"status": "online", "message": "MedFast API is running"}

# Simple health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": str(datetime.now())}

# Simplified tumor detection endpoint - returns a placeholder response
@app.post("/api/detect_tumor")
async def detect_tumor(image: UploadFile = File(...)):
    """
    Simplified tumor detection endpoint that doesn't rely on heavy ML models
    Returns a placeholder response for Vercel deployment
    """
    try:
        # Read image 
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
        
        # Create a simple response without using ML models
        # This avoids memory issues on Vercel
        response_data = {
            "tumor_detected": True,
            "message": "This is a lightweight placeholder response for Vercel deployment.",
            "note": "For full ML functionality, please run the application locally.",
            "tumors": [
                {
                    "type": "placeholder",
                    "size": "N/A",
                    "location": "N/A",
                    "confidence": "N/A"
                }
            ]
        }
        
        # Return the simple response
        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"},
            status_code=500
        )

# Simplified AI diagnosis endpoint
@app.post("/api/ai_diagnosis")
async def ai_medical_diagnosis(request: DiagnosisRequest):
    """
    Simplified AI diagnosis endpoint that doesn't use external APIs
    Returns a placeholder response
    """
    return {
        "diagnosis": "This is a simplified response for Vercel deployment. For full AI diagnosis functionality, please run the application locally.",
        "status": "ok"
    }

# Add more simplified endpoints as needed...