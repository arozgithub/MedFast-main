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
```

## Installation

**Backend Setup**  
**Clone the Repository**  
```bash
git clone https://github.com/yourusername/MedFast.git
cd MedFast/backend
```

**Create and Activate a Virtual Environment**  
**Windows:**  
```bash
python -m venv venv
venv\Scripts\activate
```
**macOS/Linux:**  
```bash
python -m venv venv
source venv/bin/activate
```

**Install Backend Dependencies**  
```bash
pip install -r requirements.txt
```
**Ensure your `requirements.txt` includes:**
- **fastapi**
- **uvicorn**
- **tensorflow**
- **keras**
- **ultralytics**
- **langchain-community (or langchain-huggingface, if applicable)**
- **faiss-cpu**
- **pdfplumber**
- **pandas**
- **tqdm**
- **imageio**



**Generate the FAISS Index (if not already present)**  
If you haven’t created your reference embeddings yet, run:
```bash
python extract_text_and_save_index.py
```
This command will create a folder (e.g., faiss_index_hp) containing the FAISS index files (index.faiss and index.pkl).


**Frontend Setup**  
**Navigate to the Frontend Directory**  
```bash
cd ../frontend
```
**Install Frontend Dependencies**  
```bash
npm install
```

**Running the Application**  
**Start the Backend**  
From the MedFast/backend directory, launch the FastAPI server:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Your backend should now be accessible at **http://127.0.0.1:8000**.


**Start the Frontend**  
In a separate terminal, navigate to the MedFast/frontend directory and start the React development server:
```bash
npm start
```
Typically, the frontend runs on **http://localhost:3000**.


## Usage

**Pneumonia Detection:**  
Upload a chest X-ray image using the XRayAnalysis component. The application processes the image and returns an annotated version indicating whether pneumonia is detected.

**Tumor Detection & Segmentation:**

**Tumor Detection:**  
Access the /detect_tumor/ endpoint to identify brain tumors in MRI scans using the YOLO model.

**Tumor Segmentation:**  
Use the /detect_tumor_h5/ endpoint to obtain a tumor mask and view the results as an animated GIF.

**Diabetes Prediction:**  
Fill out the DiabetesDetection form with the necessary predictor variables to receive a diabetes risk prediction along with an associated probability.

**Chatbot with RAG:**  
Interact with the chatbot (found in chatbot.jsx) to ask symptom-related questions. The chatbot retrieves relevant textbook references via a FAISS index and uses the Groq API to provide evidence-based recommendations.

## Deployment

**Free Deployment Options**  
**Hugging Face Spaces:**  
Deploy your FastAPI backend (or a Gradio/Streamlit app) on Hugging Face Spaces using a Docker container.

**Deta Micros:**  
Host your backend on Deta Micros for a free deployment solution.

**Docker Deployment**  
To containerize the application, create a Dockerfile in the repository root with the following content:
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
Then, build and run the container with the following commands:
```bash
docker build -t medfast-backend .
docker run -p 8000:8000 medfast-backend
```

## Environment Variables

If your application relies on environment variables (for instance, the Groq API key), create a .env file in the backend directory or configure them directly in your system. Example:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
You can use a package like python-dotenv to load these variables when needed.

## Contributing

**Contributions to MedFast are welcome! To contribute:**
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Make your changes and commit them.
- Submit a pull request for review.
- For significant changes, please open an issue first to discuss your ideas.

## License

**MedFast is licensed under the MIT License. See the LICENSE file for further details.**
