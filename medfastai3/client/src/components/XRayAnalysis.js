import React, { useState } from 'react';
import Sidebar from './Sidebar'; // Import Sidebar component
import Chatbot from './chatbot'; // Import chatbot
import './XRayAnalysis.css';  // Import the CSS file

const XRayAnalysis = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null); // Stores pneumonia detection result
  const [imageSrc, setImageSrc] = useState(null); // Optionally stores an annotated image

  // Handle file selection & processing
  const handleFileChange = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setResult("🔄 Processing...");
    setImageSrc(null); // Clear previous image

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      // Call the pneumonia detection endpoint
      const response = await fetch("http://localhost:8000/detect_pneumonia/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log("🚀 API Response:", data); // Debugging

      // After processing the API response, update result to "Processed"
      setResult("Processed");
      
      // If an annotated image is returned, display it
      if (data.image) {
        setImageSrc(`data:image/jpeg;base64,${data.image}`);
      }
    } catch (error) {
      console.error("❌ Error uploading file:", error);
      setResult("⚠️ Error processing the image.");
    }
  };

  // Toggle Sidebar
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
        <h1 className="text-2xl font-bold mb-4">
          Upload X-ray Image for Pneumonia Detection
        </h1>
        
        {/* Upload Section */}
        <div className="upload-container">
          <input
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            id="fileInput"
            onChange={handleFileChange}
          />
          <button
            onClick={() => document.getElementById('fileInput').click()}
            className="upload-button"
          >
            Upload Image
          </button>

          {/* Result Display Section */}
          {result && (
            <div className="mt-4 p-4 border rounded bg-gray-100 w-full text-left whitespace-pre-line">
              {result}
            </div>
          )}

          {/* Image Display, if provided */}
          {imageSrc && (
            <div className="mt-4">
              <h3 className="text-lg font-semibold">Processed Image</h3>
              <img
                src={imageSrc}
                alt="Pneumonia Detection"
                className="mt-2 border rounded-lg shadow-md max-w-full"
              />
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

export default XRayAnalysis;
