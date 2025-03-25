MedFast
MedFast is an integrated medical diagnosis platform that combines computer vision models and natural language processing to assist in clinical decision-making. The project features multiple diagnostic endpoints including pneumonia detection, tumor detection/segmentation, and diabetes prediction. Additionally, it includes a chatbot interface that leverages Retrieval-Augmented Generation (RAG) to provide evidence-based recommendations by retrieving relevant passages from a reference textbook.

Table of Contents
Features

Project Structure

Installation

Backend Setup

Frontend Setup

Running the Application

Deployment

Usage

Contributing

License

Features
Pneumonia Detection:
Uses a Keras/TensorFlow model to analyze chest X-ray images and classify them as "Normal" or "Pneumonia Detected." The endpoint also returns an annotated image.

Tumor Detection and Segmentation:

Tumor Detection: Uses a YOLO model (.pt file) to detect brain tumors in MRI scans, returning tumor details (type, size, location, confidence) and an annotated image.

Tumor Segmentation: Uses a Keras segmentation model (.h5 file) to generate a tumor mask and produce an animated GIF with blinking boundaries to highlight the tumor.

Diabetes Prediction:
Accepts several medical predictor variables (e.g., pregnancies, glucose, blood pressure) and uses a pre-trained model to predict diabetes risk, returning a probability and outcome.

Chatbot with RAG:
A React-based chatbot that interacts with users, maintains conversation history, and retrieves relevant references from a textbook (using FAISS embeddings). The chatbot then passes the augmented prompt (conversation history + retrieved references) to an LLM (via the Groq API) for generating evidence-based diagnosis and follow-up recommendations.

Project Structure
plaintext
Copy
MedFast/
├── backend/
│   ├── app.py                           # FastAPI application with endpoints for pneumonia, tumor, segmentation, diabetes prediction, and chatbot integration (RAG)
│   ├── extract_text_and_save_index.py   # Script to convert a textbook (in text or PDF) into embeddings and save a FAISS index (faiss_index_hp)
│   ├── model.h5                         # Keras model for pneumonia detection
│   ├── best.pt                          # YOLO model file for tumor detection
│   ├── segment_model.h5                 # Keras segmentation model for tumor segmentation
│   ├── Diabetesmodel.h5                 # Keras model for diabetes prediction
│   └── requirements.txt                 # Python dependencies for the backend
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── chatbot.jsx                  # Chatbot component with integrated RAG functionality for evidence-based responses
│   │   ├── XRayAnalysis.jsx             # Component for X-ray analysis (pneumonia detection)
│   │   ├── DiabetesDetection.jsx        # Component for diabetes prediction form
│   │   └── ...                          # Other React components and assets
│   ├── package.json                     # Frontend dependencies and scripts
│   └── ...                              # Other frontend configuration files
├── README.md                            # This file
└── LICENSE                              # MIT License file
Installation
Backend Setup
Clone the Repository:

bash
Copy
git clone https://github.com/yourusername/MedFast.git
cd MedFast/backend
Create and Activate a Virtual Environment:

bash
Copy
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Note: Your requirements.txt should include packages such as:

fastapi

uvicorn

tensorflow

keras

ultralytics

langchain-community

faiss-cpu

pdfplumber

pandas

tqdm

imageio

And others as required

Generate the FAISS Index (if not already generated):

If you haven’t built your reference index, run:

bash
Copy
python extract_text_and_save_index.py
This script will process your textbook (stored as a text file) and save a FAISS index in a folder named faiss_index_hp.

Frontend Setup
Navigate to the Frontend Directory:

bash
Copy
cd ../frontend
Install Dependencies:

bash
Copy
npm install
Running the Application
Start the Backend
From the MedFast/backend directory, run:

bash
Copy
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
Your FastAPI server should now be running at http://127.0.0.1:8000.

Start the Frontend
In a separate terminal, navigate to the MedFast/frontend directory and run:

bash
Copy
npm start
Your React development server will typically run on http://localhost:3000.

Usage
Pneumonia Detection:
Upload a chest X-ray image to /detect_pneumonia/ to classify the image and receive an annotated image.

Tumor Detection & Segmentation:

Use /detect_tumor/ for detecting tumors with bounding boxes using the YOLO model.

Use /detect_tumor_h5/ for tumor segmentation using the Keras segmentation model, which returns an animated GIF of the tumor boundaries.

Diabetes Prediction:
Fill in the medical predictor form in the frontend (DiabetesDetection component) to get a diabetes prediction.

Chatbot with RAG:
The chatbot (in chatbot.jsx) allows interactive conversation. It uses Groq to generate follow-up questions and retrieves relevant textbook references via a FAISS index. These references are then used to generate an evidence-based diagnosis when requested.

Deployment
Free Deployment Options
Hugging Face Spaces:
Deploy your FastAPI backend (or a Gradio app) on Hugging Face Spaces using a Docker container.

Deta Micros:
Host your backend on Deta Micros for free.

Docker Deployment
Create a Dockerfile in the repository root for containerization. For example:

dockerfile
Copy
FROM python:3.8-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
Then build and run the container:

bash
Copy
docker build -t medfast-backend .
docker run -p 8000:8000 medfast-backend
Environment Variables
If your project uses environment variables (e.g., for your Groq API key), create a .env file or set them in your system. For example:

bash
Copy
GROQ_API_KEY=your_groq_api_key_here
Then use a package like python-dotenv to load them in your app if needed.

Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss your ideas.

License
This project is licensed under the MIT License. See the LICENSE file for details.

