# MedFast

MedFast is an integrated medical diagnosis platform that combines advanced computer vision models and natural language processing to assist in clinical decision-making. The project includes endpoints for pneumonia detection, brain tumor detection/segmentation, and diabetes prediction. It also features an interactive chatbot that leverages Retrieval-Augmented Generation (RAG) to provide evidence-based recommendations by retrieving relevant passages from a reference textbook.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Pneumonia Detection:**  
  Uses a TensorFlow/Keras model to analyze chest X-ray images (converted to grayscale) and classify them as "Normal" or "Pneumonia Detected." The endpoint returns an annotated image.

- **Tumor Detection & Segmentation:**  
  - **Tumor Detection:** Uses a YOLO model (from a `.pt` file) to detect brain tumors in MRI scans, returning tumor details (type, size, location, confidence) along with an annotated image.
  - **Tumor Segmentation:** Uses a Keras segmentation model (from a `.h5` file) to generate a tumor mask and produce an animated GIF with blinking boundaries that highlight the tumor.

- **Diabetes Prediction:**  
  Accepts several clinical predictor variables (e.g., pregnancies, glucose, blood pressure) and uses a pre-trained model to predict diabetes risk, returning a probability and outcome.

- **Chatbot with RAG:**  
  A React-based chatbot that interacts with users by handling symptom queries, retrieving relevant textbook references (via a FAISS index built using LangChain community modules), and generating evidence-based diagnostic and treatment recommendations using the Groq API.

## Project Structure

```plaintext
MedFast/
├── backend/
│   ├── app.py                           # FastAPI application with endpoints for pneumonia, tumor, segmentation, diabetes prediction, and chatbot integration (RAG)
│   ├── extract_text_and_save_index.py   # Script to process a textbook (or text file) into embeddings and save a FAISS index (faiss_index_hp)
│   ├── model.h5                         # Keras model for pneumonia detection
│   ├── best.pt                          # YOLO model file for brain tumor detection
│   ├── segment_model.h5                 # Keras segmentation model for tumor segmentation
│   ├── Diabetesmodel.h5                 # Keras model for diabetes prediction
│   └── requirements.txt                 # Python dependencies for the backend
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── chatbot.jsx                  # Chatbot component with RAG integration for evidence-based responses
│   │   ├── XRayAnalysis.jsx             # Component for X-ray analysis (pneumonia detection)
│   │   ├── DiabetesDetection.jsx        # Component for diabetes prediction form
│   │   └── ...                          # Other React components and assets
│   ├── package.json                     # Frontend dependencies and scripts
│   └── ...                              # Other frontend configuration files
├── README.md                            # This file
└── LICENSE                              # MIT License file
````
Installation
Backend Setup
Clone the Repository:
git clone https://github.com/yourusername/MedFast.git
cd MedFast/backend
