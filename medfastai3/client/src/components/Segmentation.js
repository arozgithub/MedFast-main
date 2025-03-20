import React, { useState } from 'react';
import Sidebar from './Sidebar'; // Import Sidebar component
import Chatbot from './chatbot'; // Import chatbot
import './Segmentation.css';  // Import the CSS file

const Segmentation = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null); // Stores processing status
    const [boundaries, setBoundaries] = useState([]); // Stores tumor boundary coordinates
    const [imageSrc, setImageSrc] = useState(null); // Stores animated GIF image
  
    // Handle file selection & processing
    const handleFileChange = async (event) => {
      const selectedFile = event.target.files[0];
      if (!selectedFile) return;
  
      setFile(selectedFile);
      setResult("🔄 Processing...");
      setBoundaries([]);
      setImageSrc(null); // Clear previous image
  
      const formData = new FormData();
      formData.append("image", selectedFile);
  
      try {
        const response = await fetch("http://localhost:8000/detect_tumor_h5/", {
          method: "POST",
          body: formData,
        });
  
        const data = await response.json();
        console.log("🚀 API Response:", data); // Debugging
  
        // Update result based on API response
        if (data.tumor_detected) {
          setResult("Tumor detected and processed.");
        } else {
          setResult("No tumor detected.");
        }
        
        // Save tumor boundaries if available
        if (data.boundaries) {
          setBoundaries(data.boundaries);
        }
        
        // Display the animated GIF if available
        if (data.image) {
          setImageSrc(`data:image/gif;base64,${data.image}`);
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
          <h1 className="text-2xl font-bold mb-4">Upload an MRI Scan for Tumor Segmentation</h1>
          
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
              <div className="mt-4 p-4 border rounded bg-gray-100 w-full text-left">
                {result}
              </div>
            )}
  
            {/* Tumor Boundaries Display */}
            {boundaries.length > 0 && (
              <div className="mt-4 p-4 border rounded bg-gray-100 w-full text-left">
                <h3 className="text-lg font-semibold">Tumor Boundaries:</h3>
                {boundaries.map((boundary, index) => (
                  <div key={index}>
                    <strong>Region {index + 1}:</strong> {JSON.stringify(boundary)}
                  </div>
                ))}
              </div>
            )}
  
            {/* Image Display */}
            {imageSrc && (
              <div className="mt-4">
                <h3 className="text-lg font-semibold">Processed MRI Scan</h3>
                <img
                  src={imageSrc}
                  alt="Tumor Segmentation"
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
  
export default Segmentation;
