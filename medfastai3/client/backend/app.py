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
GROQ_API_KEY = "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"

# Initialize separate Groq clients for diagnosis and follow-up
groq_diagnosis_client = Groq(api_key=GROQ_API_KEY)
groq_questions_client = Groq(api_key=GROQ_API_KEY)

# Request Models
class DiagnosisRequest(BaseModel):
    conversation_history: str

class FollowUpRequest(BaseModel):
    conversation_history: str


# class DiagnosisRequest_doc(BaseModel):
#     conversation_history_doc: str

# class FollowUpRequest_doc(BaseModel):
#     conversation_history_doc: str

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
            "You are an empathetic and context-aware AI assistant responsible for generating follow-up questions based on a patient's reported symptoms and conversation history. "
            "Your task is to carefully review all prior inputs and extract specific details about the patient's symptoms. Then, craft a follow-up question that directly addresses those details, "
            "aiming to gather additional clarity on the context, severity, or triggers of a particular symptom.\n\n"
            
            "1️⃣ **Deep Context Analysis**: Analyze the conversation history to identify any symptom or detail that needs further exploration. "
            "For instance, if the patient mentioned 'persistent headaches' or 'a sharp pain in the left side,' ensure your question specifically refers to that detail.\n\n"
            
            "2️⃣ **Tailored Question Crafting**: Formulate your follow-up question by referencing the exact symptom or context mentioned in the conversation. "
            "For example, instead of asking, 'Can you tell me more about your headache?', ask 'You mentioned experiencing a sharp pain on the left side of your head—can you describe how this pain changes with different activities?'\n\n"
            
            "3️⃣ **Engaging and Specific Queries**: Your follow-up question should be concise, engaging, and designed to elicit more detailed information. "
            "Ensure that your question feels natural and directly connected to what the patient has already shared.\n\n"
            
            "Bot response format:\n"
            "- Bot: [A context-rich follow-up question addressing a specific detail from the conversation history]\n"
            "- Only one follow-up question per response, ensuring it is clear and focused."
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
    

def query_groq_medical_diagnosis_doc(conversation_history):
    try:
        messages = [
    {
        "role": "system",
        "content": (
            "You are an AI medical assistant specializing in brain tumor diagnosis and treatment. "
            "You analyze MRI scans using YOLO model outputs, evaluate patient symptoms, and provide detailed, step-by-step professional medical insights. "
            "Keep the context of previous inputs to ensure accurate follow-up questions and responses.\n\n"

            "1️⃣ **Initial Diagnosis (Display Once)**: After MRI analysis, provide the following details: tumor type, tumor location, tumor size, and YOLO model confidence. "
            "If a tumor is detected, display 'Tumor Detected: Yes' along with a detailed tumor list including all relevant parameters. "
            "If no tumor is detected, display 'Tumor Detected: No' and proceed with a symptom-based analysis. "
            "Example Output: 'Tumor Detected: Yes, Tumor List: Type: meningioma, Location: Left, Size: 40.55mm x 38.76mm, Confidence: 0.37%'\n\n"

            "2️⃣ **Symptoms Analysis & Diagnostic Tests**: Explain how the tumor correlates with the reported symptoms and provide recommendations for further diagnostic tests. "
            "These tests might include CT scans, biopsies, blood tests, PET scans, or any other relevant investigations to confirm or further evaluate the condition.\n\n"

            "3️⃣ **Treatment Plan & Medication Prescription**: Provide a comprehensive treatment plan tailored to the tumor type and patient symptoms. "
            "Include recommendations for surgical intervention, radiation therapy, chemotherapy, or referrals to specialists. "
            "If medications are indicated, list specific prescription names (e.g., Temozolomide, Bevacizumab) along with dosage guidelines or administration instructions where applicable."
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
        
    
def query_groq_follow_up_questions_doc(conversation_history):
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant responsible for generating follow-up questions for a doctor based on their findings from an MRI scan and the patient's reported symptoms. "
                    "You will use the conversation history—which includes the doctor's detailed MRI observations and the patient's symptoms—to ask targeted, clarifying questions. "
                    "Your questions should help resolve any ambiguities in the MRI findings and further explore the patient's symptoms to support a more precise diagnosis.\n\n"
                    
                    "1️⃣ **Doctor Consultation**: Ask context-aware follow-up questions that specifically address the MRI findings and the patient’s symptoms. "
                    "For example, if the doctor mentioned an irregular lesion or a specific anomaly in the MRI, and the patient reported related discomfort or other symptoms, your question should probe further into those details. "
                    "Example: 'Can you elaborate on the irregular shape observed in the left hemisphere and confirm if the patient experiences correlated pain in that region?'\n\n"
                    
                    "2️⃣ **Iterative Clarification**: Ensure that you ask only one focused follow-up question per response. "
                    "Each question should be concise, directly related to the previously provided information, and designed to gather additional details until the doctor indicates no further observations are needed.\n\n"
                    
                    "Bot response format:\n"
                    "- Bot: [Follow-up question targeting MRI findings and patient symptoms]\n"
                    "- Only one follow-up question per response, ensuring it is context-aware and concise."
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



@app.post("/ai_diagnosis_doc/")
async def ai_medical_diagnosis_doc(request: DiagnosisRequest):
    diagnosis= query_groq_medical_diagnosis_doc(request.conversation_history)
    return {"diagnosis": diagnosis}

# Endpoint: AI-Generated Follow-Up Questions
@app.post("/ai_followup_doc/")
async def ai_followup_questions_doc(request: FollowUpRequest):
    follow_up = query_groq_follow_up_questions_doc(request.conversation_history)
    return {"follow_up_question": follow_up}


# To run the app, use: uvicorn app:app --reload
