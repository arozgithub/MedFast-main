import React, { useState } from "react";
import { Send, X, MessageCircle, Plus, Activity } from "lucide-react";
import "./chatbot.css";

const Chatbot = () => {
  const [messages, setMessages] = useState([
    { text: "Hi! How can I assist with your diagnosis?", sender: "bot" },
  ]);
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [file, setFile] = useState(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: "user" };
    setMessages([...messages, userMessage]);
    setInput("");

    try {
      const response = await fetch("http://localhost:8000/ai_followup/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_history: input }),
      });

      const data = await response.json();
      const botMessage = { text: data.follow_up_question, sender: "bot" };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
    }
  };

  const handleFileChange = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
  
    setFile(selectedFile);
    setMessages([...messages, { text: `📎 ${selectedFile.name}`, sender: "user" }]);
  
    const formData = new FormData();
    formData.append("image", selectedFile);
  
    try {
      const response = await fetch("http://localhost:8000/detect_tumor/", {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      console.log("🔍 Tumor Detection API Response:", data); // Debugging
  
      let botMessage;
      if (data.tumor_detected && data.tumors.length > 0) {
        // Extracting tumor details
        const tumorsInfo = data.tumors
          .map(
            (tumor, index) =>
              `🧠 **Tumor ${index + 1}**\n🔬 **Type**: ${tumor.tumor_type}\n📏 **Size**: ${tumor.size}\n📍 **Location**: ${tumor.location}\n💡 **Confidence**: ${tumor.confidence}`
          )
          .join("\n\n");
  
        botMessage = { text: tumorsInfo, sender: "bot" };
      } else {
        botMessage = { text: "No tumor detected.", sender: "bot" };
      }
  
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error uploading file:", error);
      setMessages((prev) => [...prev, { text: "⚠️ Error processing the image.", sender: "bot" }]);
    }
  };
  
  const diagnose = async () => {
    setMessages([...messages, { text: "Diagnosing your condition...", sender: "bot" }]);

    try {
      const response = await fetch("http://localhost:8000/ai_diagnosis/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_history: input }),
      });

      const data = await response.json();
      setMessages((prev) => [...prev, { text: data.diagnosis, sender: "bot" }]);
    } catch (error) {
      console.error("Error diagnosing:", error);
    }
  };

  return (
    <div className={`chatbot-container ${isOpen ? "open" : ""}`}>
      {isOpen ? (
        <div className="chatbot-box">
          <div className="chatbot-header">
            <span>AI Medical Assistant</span>
            <X className="close-btn" onClick={() => setIsOpen(false)} />
          </div>
          <div className="chatbot-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.sender}`}>
                {msg.text}
              </div>
            ))}
          </div>
          <div className="chatbot-input">
            <input
              type="text"
              placeholder="Describe your symptoms..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button onClick={sendMessage}>
              <Send size={18} />
            </button>
            <label className="upload-btn">
              <Plus size={18} />
              <input type="file" onChange={handleFileChange} style={{ display: "none" }} />
            </label>
            <button className="diagnose-btn" onClick={diagnose}>
              <Activity size={18} />
            </button>
          </div>
        </div>
      ) : (
        <div className="chatbot-icon" onClick={() => setIsOpen(true)}>
          <MessageCircle size={24} />
        </div>
      )}
    </div>
  );
};

export default Chatbot;
