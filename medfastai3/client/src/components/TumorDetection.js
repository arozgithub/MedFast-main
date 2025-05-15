import React, { useState } from 'react';
import Sidebar from './Sidebar'; // Import Sidebar component
import Chatbot from './chatbot'; // Import chatbot
import './TumorDetection.css';  // Import the CSS file

const TumorDetection = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null); // Stores tumor detection results
    const [imageSrc, setImageSrc] = useState(null); // Stores processed image

    // 📂 Handle file selection & processing
    const handleFileChange = async (event) => {
        const selectedFile = event.target.files[0];
        if (!selectedFile) return;

        setFile(selectedFile);
        setResult("🔄 Processing...");
        setImageSrc(null); // Clear previous image

        const formData = new FormData();
        formData.append("image", selectedFile);

        try {
            const response = await fetch("http://localhost:8000/detect_tumor/", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            console.log("🚀 API Response:", data); // Debugging

            if (data.tumor_detected && data.tumors.length > 0) {
                // Extracting tumor details
                const tumorsInfo = data.tumors
                    .map(
                        (tumor, index) =>
                            `🧠 Tumor ${index + 1}\n🔬 Type: ${tumor.type}\n📏 Size: ${tumor.size}\n📍 Location: ${tumor.location}\n💡 Confidence: ${tumor.confidence}`
                    )
                    .join("\n\n");

                setResult(tumorsInfo);
                setImageSrc(`data:image/jpeg;base64,${data.image}`); // Convert Base64 to Image URL
            } else {
                setResult("✅ No tumor detected.");
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
                <h1 className="text-2xl font-bold mb-4">Upload an MRI Scan</h1>
                
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

                    {/* Image Display */}
                    {imageSrc && (
                        <div className="mt-4">
                            <h3 className="text-lg font-semibold">Processed MRI Scan</h3>
                            <img
                                src={imageSrc}
                                alt="Tumor Detection"
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

export default TumorDetection;
