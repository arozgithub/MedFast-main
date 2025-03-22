from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from groq import Groq  # Use `Groq` as needed
import io
import asyncio
from fastapi.responses import JSONResponse
import base64
from PIL import Image
import logging
import numpy as np
import cv2
import imageio
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from keras.layers import InputLayer as KerasInputLayer
from tensorflow.keras.mixed_precision import Policy
from langchain_community.vectorstores import FAISS
# extract_text_and_save_index.py

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import NLTKTextSplitter  # optional alternative

# For prompts and chains (if needed)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



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

rag_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Load the FAISS index from your saved folder "faiss_index_hp"
vectordb = FAISS.load_local("faiss_index_hp", rag_embeddings, allow_dangerous_deserialization=True)

print("Reference FAISS index loaded.")
# ---------------------
# Pneumonia Detection Setup
# ---------------------

# Define a custom InputLayer to handle the batch_shape keyword if needed
class CustomInputLayer(KerasInputLayer):
    def __init__(self, **kwargs):
        # Replace 'batch_shape' with 'batch_input_shape' if present
        if "batch_shape" in kwargs:
            kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
        super(CustomInputLayer, self).__init__(**kwargs)

# Register custom object for the dtype policy
custom_objects = {'InputLayer': CustomInputLayer, 'DTypePolicy': Policy}

MODEL_PATH = 'model.h5'

# Load the model using the custom objects
try:
    model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

# Define a function to preprocess an image file for prediction (using file path)
def preprocess_image(image_path, img_size=150):
    """
    Preprocess the input image from a file path:
      - Reads the image in grayscale.
      - Resizes it to (img_size, img_size).
      - Normalizes pixel values to the range [0, 1].
      - Reshapes it to (1, img_size, img_size, 1) for model prediction.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}. Check the file path.")
    
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype("float32") / 255.0
    image = image.reshape(1, img_size, img_size, 1)
    return image

# ---------------------
# FastAPI Endpoints
# ---------------------

@app.post("/detect_pneumonia/")
async def detect_pneumonia(image: UploadFile = File(...)):
    try:
        # Read the uploaded image from the request
        img = Image.open(io.BytesIO(await image.read()))
        # Convert to grayscale (since our model expects grayscale images)
        img = img.convert("L")
        
        # Resize the image to match the model's input size (150x150)
        img = img.resize((150, 150))
        
        # Convert image to numpy array, normalize and reshape to (1, 150, 150, 1)
        img_array = np.array(img, dtype="float32") / 255.0
        img_array = img_array.reshape(1, 150, 150, 1)
        
        # Run the model prediction
        prediction = model.predict(img_array)
        # For binary classification using a sigmoid activation:
        # If prediction > 0.5 then assume label "Normal", else "Pneumonia Detected"
        pneumonia_probability = float(prediction[0][0])
        result = "Normal" if pneumonia_probability > 0.5 else "Pneumonia Detected"
        
        # --- Annotate the image ---
        # Convert the grayscale image (PIL Image) to a NumPy array for OpenCV processing
        annotated_img = np.array(img)
        # Convert grayscale to BGR (so that colored text can be drawn)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_GRAY2BGR)
        # Define text parameters
        text = result
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0)  # green text
        thickness = 2
        # Put the result text on the image
        cv2.putText(annotated_img, text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        # Encode the annotated image as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        response_data = {
            "result": result,
            "pneumonia_probability": pneumonia_probability,
            "image": encoded_image
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        logging.error(f"Error in detect_pneumonia: {str(e)}")
        return JSONResponse(content={"error": "Failed to detect pneumonia"}, status_code=500)



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
        # Retrieve top 3 reference passages using the FAISS index (vectordb)
        docs = vectordb.similarity_search(conversation_history, k=3)
        refs = [doc.page_content for doc in docs]

        # Print retrieved references for debugging
        print("Retrieved references:")
        for i, ref in enumerate(refs):
            print(f"Reference {i+1}: {ref}")

        # Build a detailed system message with instructions and include references information
        system_message = (
            "You are an AI medical assistant specializing in brain tumor diagnosis and treatment. "
            "You analyze MRI scans using YOLO model outputs, evaluate patient symptoms, and provide detailed, step-by-step professional medical insights. "
            "Keep the context of previous inputs to ensure accurate follow-up questions and responses.\n\n"
            "1️⃣ **Initial Diagnosis (Display Once)**: After MRI analysis, provide the following details: tumor type, tumor location, tumor size, and YOLO model confidence. "
            "If a tumor is detected, display 'Tumor Detected: Yes' along with a detailed tumor list including all relevant parameters. "
            "If no tumor is detected, display 'Tumor Detected: No' and proceed with a symptom-based analysis. "
            "Example Output: 'Tumor Detected: Yes, Tumor List: Type: meningioma, Location: Left, Size: 40.55mm x 38.76mm, Confidence: 0.37%'\n\n"
            "2️⃣ **Symptoms Analysis & Diagnostic Tests**: Explain how the tumor correlates with the reported symptoms and provide recommendations for further diagnostic tests. "
            "These tests might include CT scans, biopsies, blood tests, PET scans, or any other relevant investigations.\n\n"
            "3️⃣ **Treatment Plan & Medication Prescription**: Provide a comprehensive treatment plan tailored to the tumor type and patient symptoms. "
            "Include recommendations for surgical intervention, radiation therapy, chemotherapy, or referrals to specialists. "
            "If medications are indicated, list specific prescription names (e.g., Temozolomide, Bevacizumab) along with dosage guidelines or administration instructions.\n\n"
            "At the end of your answer, please clearly list the references used (i.e. the following retrieved passages):\n"
        )
        # Append each retrieved reference to the system message
        for i, ref in enumerate(refs):
            system_message += f"Reference {i+1}: {ref}\n\n"

        # Now, include the conversation history as a user message
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Conversation History:\n" + conversation_history}
        ]

        # Call the Groq API for diagnosis using the augmented prompt
        response = groq_diagnosis_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile"
        )
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


#----------------------------------------------------------------------------
#segementation:
TUMOR_MODEL_PATH = "C:/MedFast-main/medfastai3/client/backend/segment_model.h5"
tumor_model = load_model(TUMOR_MODEL_PATH, compile=False)

print(f"Tumor segmentation model loaded: {tumor_model}")

def preprocess_tumor_image(image, target_size=(256, 256)):
    """
    Preprocess the input image:
      - Ensure the image is RGB.
      - Resize the image to the target size.
      - Normalize pixel values to [0,1].
      - Expand dimensions for model prediction.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image, dtype="float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/detect_tumor_h5/")
async def detect_tumor_h5(image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        img = Image.open(io.BytesIO(await image.read()))
        
        # Preprocess the image (adjust target_size as needed)
        processed_img = preprocess_tumor_image(img, target_size=(256, 256))
        
        loop = asyncio.get_event_loop()
        # Run the heavy segmentation task in a background thread
        prediction = await loop.run_in_executor(None, lambda: tumor_model.predict(processed_img))
        mask = prediction[0, :, :, 0]  # Extract the mask
        
        # Threshold the mask to obtain a binary mask (adjust threshold if necessary)
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        
        # Extract tumor boundaries using contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tumor_boundaries = []
        for contour in contours:
            points = contour.squeeze().tolist()
            if isinstance(points[0], int):
                points = [points]
            tumor_boundaries.append(points)
        
        # Resize the original image for visualization (to match target_size)
        img_resized = img.resize((256, 256))
        img_np = np.array(img_resized)
        
        # Create an overlay: for pixels where the mask is positive, paint red
        overlay = img_np.copy()
        overlay[mask_binary == 255] = [255, 0, 0]
        
        # Blend the original image and the overlay (base frame without boundaries)
        alpha = 0.5
        annotated_no_contour = cv2.addWeighted(img_np, 1 - alpha, overlay, alpha, 0)
        # Create a frame with boundaries: copy the base frame and draw contours in green
        annotated_with_contour = annotated_no_contour.copy()
        cv2.drawContours(annotated_with_contour, contours, -1, (0, 255, 0), 2)
        
        # Create an animated GIF from two frames (blinking effect, looping indefinitely)
        frames = [annotated_with_contour, annotated_no_contour]
        gif_buffer = io.BytesIO()
        import imageio
        imageio.mimsave(gif_buffer, frames, format='GIF', duration=0.5, loop=0)
        gif_buffer.seek(0)
        encoded_gif = base64.b64encode(gif_buffer.read()).decode("utf-8")
        
        response_data = {
            "tumor_detected": bool(np.any(mask_binary)),
            "boundaries": tumor_boundaries,
            "image": encoded_gif
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        logging.error(f"Error in detect_tumor_h5: {str(e)}")
        return JSONResponse(content={"error": "Failed to detect tumor using h5 model"}, status_code=500)

    ####################################
# Diabetes Prediction Endpoint     #
####################################

# Define a Pydantic model for the diabetes input data
class DiabetesInput(BaseModel):
    pregnancies: float
    glucose: float
    bloodPressure: float
    skinThickness: float
    insulin: float
    BMI: float
    diabetesPedigreeFunction: float
    age: float

# Load the diabetes model from the h5 file
DIABETES_MODEL_PATH = "Diabetesmodel.h5"  # Update this path if needed
diabetes_model = load_model(DIABETES_MODEL_PATH, compile=False)
diabetes_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Diabetes model loaded:")
# diabetes_model.summary()

@app.post("/predict_diabetes/")
async def predict_diabetes(input_data: DiabetesInput):
    try:
        # Convert the input data to a numpy array in the correct order
        data_array = np.array([
            input_data.pregnancies,
            input_data.glucose,
            input_data.bloodPressure,
            input_data.skinThickness,
            input_data.insulin,
            input_data.BMI,
            input_data.diabetesPedigreeFunction,
            input_data.age
        ], dtype="float32")
        # Reshape to (1,8) instead of (1,1,8)
        data_array = data_array.reshape(1, 8)
        
        predictions = diabetes_model.predict(data_array)
        # Assuming the model outputs a single probability per sample
        prediction_probability = float(predictions[0][0])
        outcome = 1 if prediction_probability > 0.5 else 0
        return JSONResponse(content={
            "prediction_probability": prediction_probability,
            "predicted_outcome": outcome
        })
    except Exception as e:
        logging.error(f"Error in predict_diabetes: {str(e)}")
        return JSONResponse(content={"error": "Failed to predict diabetes outcome"}, status_code=500)


# To run the app, use: uvicorn app:app --reload
