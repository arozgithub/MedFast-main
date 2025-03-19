from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from groq import Groq  # Use `Groq` as needed
import io
from fastapi.responses import JSONResponse
import base64
from PIL import Image
import logging
import numpy as np
import cv2
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Enable CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model (update the model path as necessary)
MODEL_PATH = "best.pt"
yolo_model = YOLO(MODEL_PATH)
print(f"Model loaded: {yolo_model}")

# Define class names (matching the ipynb)
CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}

# Groq API Key (use environment variables in production)
GROQ_API_KEY = "your_groq_api_key_here"

# Initialize separate Groq clients for diagnosis and follow-up
groq_diagnosis_client = Groq(api_key=GROQ_API_KEY)
groq_questions_client = Groq(api_key=GROQ_API_KEY)

# Request Models
class DiagnosisRequest(BaseModel):
    conversation_history: str

class FollowUpRequest(BaseModel):
    conversation_history: str

def get_tumor_location(x_center, img_width):
    """Determine tumor location based on x_center relative to image width."""
    if x_center < img_width * 0.33:
        return "left hemisphere"
    elif x_center > img_width * 0.66:
        return "right hemisphere"
    else:
        return "central region"

# AI Medical Diagnosis using Groq
def query_groq_medical_diagnosis(conversation_history):
    try:
        messages = [
    {
        "role": "system",
        "content": (
            "You are an AI medical assistant specializing in brain tumor diagnosis and treatment. "
            "You analyze MRI scans using YOLO model outputs, review patient symptoms, and provide detailed, professional medical insights. "
            "Keep the context of previous inputs to ensure appropriate follow-up questions and responses.\n\n"
            "1️⃣ **Initial Diagnosis (Display Once)**: After analyzing the MRI scan, provide the following details: tumor type, tumor location, tumor size, and YOLO model confidence. "
            "If a tumor is detected, display 'Tumor Detected: Yes' along with a list of tumor details. "
            "If no tumor is detected, display 'Tumor Detected: No' and proceed with a symptom-based analysis.\n\n"
            "2️⃣ **Symptoms Analysis**: Analyze and explain how the tumor (if detected) relates to the patient's provided symptoms, referencing the conversation history.\n\n"
            "3️⃣ **Treatment Plan**: Provide a comprehensive treatment plan. Include options such as surgery, radiation, chemotherapy, or referrals. "
            "If medications are recommended, list specific medication prescription names and details (for example, dosage and administration instructions) that are typically used in such cases.\n\n"
            "4️⃣ **Step-by-Step Explanation**: Clearly outline the diagnostic process in steps, including how the MRI analysis, symptom evaluation, and treatment recommendations were derived. "
            "Each step should be clearly delineated and supported by relevant medical reasoning.\n\n"
            "Ensure that your response is precise, detailed, and provides clear guidance at every step of the diagnosis and treatment planning process."
        )
    },
    {
        "role": "user",
        "content": f"Conversation History: {conversation_history}"
    }
]

        response = groq_diagnosis_client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in query_groq_medical_diagnosis: {str(e)}")
        return {"error": "Failed to generate diagnosis"}

# AI Follow-Up Question Generator using Groq
def query_groq_follow_up_questions(conversation_history):
    try:
        messages = [
          {
              "role": "system",
              "content": (
                  "You are an AI assistant responsible for generating follow-up questions for a patient based on their provided symptoms. "
                  "You will use the conversation history and previously provided symptoms to ask further questions. "
                  "Ensure context from the patient's earlier responses is maintained while asking relevant questions.\n\n"

                  "1️⃣ **Symptom Evaluation**: Use the patient's previously provided symptoms to generate context-aware follow-up questions. "
                  "- Ensure you only ask one follow-up question per response, and the question should be focused on gathering more details about a specific symptom.\n"
                  "- Example Output: Can you describe more about your headache?\n\n"

                  "2️⃣ **Follow-Up Questions**: After each user response, continue the conversation by asking relevant questions based on the newly provided symptoms, until the patient indicates no more symptoms to discuss.\n\n"

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
        response = groq_questions_client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in query_groq_follow_up_questions: {str(e)}")
        return {"error": "Failed to generate follow-up question"}

# YOLO-based MRI Tumor Detection Endpoint
@app.post("/detect_tumor/")
async def detect_tumor(image: UploadFile = File(...)):
    try:
        # Read image and convert to RGB
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
        img_np = np.array(img)
        print(f"Image shape: {img_np.shape}")

        # Perform tumor detection using YOLO
        results = yolo_model(img_np, imgsz=1024, conf=0.2)

        tumors = []
        pixel_to_mm = 0.5  # Adjust as necessary
        tumor_detected = False

        for i, box in enumerate(results[0].boxes.xywh.cpu().numpy()):
            class_id = int(results[0].boxes.cls[i])
            confidence = results[0].boxes.conf[i].item()
            tumor_label = CLASS_NAMES.get(class_id, "Unknown")

            if tumor_label == "notumor":
                continue  # Skip if no tumor is detected

            tumor_detected = True
            tumor_info = {
                "type": tumor_label,
                "size": f'{box[2] * pixel_to_mm:.2f}mm x {box[3] * pixel_to_mm:.2f}mm',
                "location": get_tumor_location(box[0], img_np.shape[1]),
                "confidence": f'{confidence:.2f}'
            }
            tumors.append(tumor_info)

            # Draw bounding box (using red color as in ipynb)
            x_center, y_center, width, height = box
            x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
            x2, y2 = int(x_center + width / 2), int(y_center + height / 2)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Convert the modified image back to a PIL image and encode to Base64
        modified_image = Image.fromarray(img_np)
        buffered = io.BytesIO()
        modified_image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response_data = {
            "tumor_detected": tumor_detected,
            "tumors": tumors,
            "image": encoded_image
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

# To run the app, use: uvicorn app:app --reload
