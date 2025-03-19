from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from groq import Groq  # Use `Groq` instead of `GroqClient`
import io
from fastapi.responses import JSONResponse
import base64
from PIL import Image
import logging
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Enable CORS (Adjust `allow_origins` for better security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model (Replace with actual trained model path)
yolo_model = YOLO(r"C:\MedFast-main\medfastai3\client\backend\best.pt")  # Raw string format
print(f"Model loaded: {yolo_model}")

# Class names mapping (Update based on trained model)
CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}

# Groq API Key (Use environment variables in production)
GROQ_API_KEY = "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"

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# Request Models
class DiagnosisRequest(BaseModel):
    conversation_history: str

class FollowUpRequest(BaseModel):
    conversation_history: str

def get_tumor_location(x_center, img_width):
    """Estimate tumor location based on the image width."""
    if x_center < img_width / 3:
        return "Left"
    elif x_center > 2 * img_width / 3:
        return "Right"
    else:
        return "Center"

# AI Medical Diagnosis using Groq
def query_groq_medical_diagnosis(conversation_history):
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI medical assistant specializing in brain tumor diagnosis and treatment. "
                    "You analyze MRI scans using YOLO model outputs, patient symptoms, and provide professional medical insights. "
                    "Keep the context of previous inputs to ensure appropriate follow-up questions and responses.\n\n"
                    "1️⃣ **Initial Diagnosis (Display Once)**: After MRI analysis, provide tumor type, location, size, and YOLO model confidence.\n"
                    "2️⃣ **Symptoms Analysis**: Explain how the tumor relates to the provided symptoms.\n"
                    "3️⃣ **Treatment Plan**: Provide appropriate treatment based on the symptoms and tumor type (e.g., surgery, medication, or referrals).\n"
                )
            },
            {
                "role": "user",
                "content": f"Conversation History: {conversation_history}"
            }
        ]
        response = groq_client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in query_groq_medical_diagnosis: {str(e)}")
        return {"error": "Failed to generate diagnosis"}

# AI Follow-Up Question Generator
def query_groq_follow_up_questions(conversation_history):
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant responsible for generating follow-up questions for a patient based on their provided symptoms. "
                    "You will use the conversation history and previously provided symptoms to ask further questions. "
                    "Ensure context from the patient's earlier responses is maintained while asking relevant questions.\n\n"
                    "1️⃣ **Symptom Evaluation**: Generate context-aware follow-up questions based on symptoms.\n"
                    "2️⃣ **Follow-Up Questions**: Continue conversation by asking relevant questions until all symptoms are covered.\n\n"
                    "Bot response format:\n"
                    "- Bot: [follow-up question based on symptom]\n"
                    "- Only one follow-up question per response, context-aware and concise."
                )
            },
            {
                "role": "user",
                "content": f"Conversation History: {conversation_history}"
            }
        ]
        response = groq_client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in query_groq_follow_up_questions: {str(e)}")
        return {"error": "Failed to generate follow-up question"}



# YOLO-based MRI Tumor Detection
import json

@app.post("/detect_tumor/")
async def detect_tumor(image: UploadFile = File(...)):
    try:
        # Read image
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
        
        img_np = np.array(img)
        print(f"Image shape: {img_np.shape}")  # Check the shape of the image



        # Perform tumor detection using YOLO
        results = yolo_model(img_np, imgsz=1024, conf=0.2)

        tumors = []
        pixel_to_mm = 0.5  # Adjust based on dataset calibration
        tumor_detected = False

        # Draw bounding boxes
        for i, box in enumerate(results[0].boxes.xywh.cpu().numpy()):
            class_id = int(results[0].boxes.cls[i])
            confidence = results[0].boxes.conf[i].item()
            tumor_label = CLASS_NAMES.get(class_id, "Unknown")

            if tumor_label == "notumor":
                continue  # Ignore cases where no tumor is detected

            tumor_detected = True
            tumor_info = {
                'tumor_type': tumor_label,
                'size': f'{box[2] * pixel_to_mm:.2f}mm x {box[3] * pixel_to_mm:.2f}mm',
                'location': get_tumor_location(box[0], img_np.shape[1]),
                'confidence': f'{confidence:.2f}'
            }
            tumors.append(tumor_info)

            # Get bounding box coordinates
            x_center, y_center, width, height = box
            x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
            x2, y2 = int(x_center + width / 2), int(y_center + height / 2)

            # Draw rectangle on the image
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert the modified image back to a PIL image
        modified_image = Image.fromarray(img_np)

        # Convert the image to Base64 for response
        buffered = io.BytesIO()
        modified_image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response_data = {
            "tumor_detected": tumor_detected,
            "tumors": tumors,
            "image": encoded_image  # Base64 encoded image
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logging.error(f"Error in detect_tumor: {str(e)}")
        return {"error": "Failed to detect tumor"}

# Endpoint: AI-Powered Medical Diagnosis
@app.post("/ai_diagnosis/")
async def ai_medical_diagnosis(request: DiagnosisRequest):
    diagnosis = query_groq_medical_diagnosis(request.conversation_history)
    return {"diagnosis": diagnosis}

# Endpoint: AI-Generated Follow-Up Questions
@app.post("/ai_followup/")
async def ai_followup_questions(request: FollowUpRequest):
    follow_up = query_groq_follow_up_questions(request.conversation_history)
    return {"follow_up_question": follow_up}

# Run the FastAPI app using:
# uvicorn app:app --reload
