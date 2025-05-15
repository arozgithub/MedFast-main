import React, { useState } from 'react';
import Sidebar from './Sidebar'; // Import Sidebar component
import Chatbot from './chatbot'; // Import chatbot
import './DiabetesDetection.css';  // Import the CSS file

const DiabetesDetection = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  // State variables for each predictor
  const [pregnancies, setPregnancies] = useState('');
  const [glucose, setGlucose] = useState('');
  const [bloodPressure, setBloodPressure] = useState('');
  const [skinThickness, setSkinThickness] = useState('');
  const [insulin, setInsulin] = useState('');
  const [BMI, setBMI] = useState('');
  const [diabetesPedigreeFunction, setDiabetesPedigreeFunction] = useState('');
  const [age, setAge] = useState('');
  
  const [result, setResult] = useState(null); // Stores prediction result

  // Handle form submission for diabetes prediction
  const handleSubmit = async (event) => {
    event.preventDefault();
    setResult("🔄 Processing...");
    
    const payload = {
      pregnancies: parseFloat(pregnancies),
      glucose: parseFloat(glucose),
      bloodPressure: parseFloat(bloodPressure),
      skinThickness: parseFloat(skinThickness),
      insulin: parseFloat(insulin),
      BMI: parseFloat(BMI),
      diabetesPedigreeFunction: parseFloat(diabetesPedigreeFunction),
      age: parseFloat(age)
    };

    try {
      const response = await fetch("http://localhost:8000/predict_diabetes/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      console.log("🚀 API Response:", data); // Debugging
      
      // Map the predicted outcome: 1 means diabetic, else not diabetic
      const outcomeText = data.predicted_outcome === 1 ? "Diabetic" : "Not Diabetic";
      setResult(
        `Prediction: ${outcomeText} `
        // (Probability: ${(data.prediction_probability * 100).toFixed(2)}%)
      );
    } catch (error) {
      console.error("❌ Error:", error);
      setResult("⚠️ Error processing the data.");
    }
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="segmentation-container">
      {/* Sidebar */}
      <Sidebar isOpen={isSidebarOpen} />
      <button 
        onClick={toggleSidebar} 
        className={`sidebar-toggle ${isSidebarOpen ? 'open' : 'closed'}`}
      >
        {isSidebarOpen ? '◄' : '►'}
      </button>

      {/* Main Content */}
      <div className={`segmentation-content ${isSidebarOpen ? 'shifted' : ''}`}>
        <h1 className="text-2xl font-bold mb-4">Diabetes Prediction</h1>
        <div className="upload-container">
          <form onSubmit={handleSubmit} className="diabetes-form">
            <div className="form-group">
              <label>Pregnancies:</label>
              <input 
                type="number" 
                value={pregnancies} 
                onChange={(e) => setPregnancies(e.target.value)} 
                required 
              />
            </div>
            <div className="form-group">
              <label>Glucose:</label>
              <input 
                type="number" 
                value={glucose} 
                onChange={(e) => setGlucose(e.target.value)} 
                required 
              />
            </div>
            <div className="form-group">
              <label>Blood Pressure:</label>
              <input 
                type="number" 
                value={bloodPressure} 
                onChange={(e) => setBloodPressure(e.target.value)} 
                required 
              />
            </div>
            <div className="form-group">
              <label>Skin Thickness:</label>
              <input 
                type="number" 
                value={skinThickness} 
                onChange={(e) => setSkinThickness(e.target.value)} 
                required 
              />
            </div>
            <div className="form-group">
              <label>Insulin:</label>
              <input 
                type="number" 
                value={insulin} 
                onChange={(e) => setInsulin(e.target.value)} 
                required 
              />
            </div>
            <div className="form-group">
              <label>BMI:</label>
              <input 
                type="number" 
                step="0.1" 
                value={BMI} 
                onChange={(e) => setBMI(e.target.value)} 
                required 
              />
            </div>
            <div className="form-group">
              <label>Diabetes Pedigree Function:</label>
              <input 
                type="number" 
                step="0.01" 
                value={diabetesPedigreeFunction} 
                onChange={(e) => setDiabetesPedigreeFunction(e.target.value)} 
                required 
              />
            </div>
            <div className="form-group">
              <label>Age:</label>
              <input 
                type="number" 
                value={age} 
                onChange={(e) => setAge(e.target.value)} 
                required 
              />
            </div>
            <button type="submit" className="submit-button">Predict</button>
          </form>

          {/* Result Display Section */}
          {result && (
            <div className="result-display">
              {result}
            </div>
          )}
        </div>
      </div>

      {/* Chatbot Section */}
      <div className="w-1/2 p-6 flex flex-col bg-gray-200 relative">
        <Chatbot />
      </div>
    </div>
  );
};

export default DiabetesDetection;
