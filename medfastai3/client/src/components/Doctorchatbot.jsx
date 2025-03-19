import React, { useState } from "react";
import { Send, X, MessageCircle, Plus, Activity } from "lucide-react";
import "./chatbot.css";

const DoctorChatbot = () => {
  const [messages, setMessages] = useState([
    {
      text: "Welcome Doctor. Please upload the MRI image and share your observations (comma-separated) to begin the consultation.",
      sender: "bot",
    },
  ]);
  const [conversationHistory, setConversationHistory] = useState("");
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [file, setFile] = useState(null);

  // Append new messages to the conversation history
  const updateHistory = (sender, text) => {
    const updatedHistory = conversationHistory + `\n${sender}: ${text}`;
    setConversationHistory(updatedHistory);
  };

  // Send doctor's message and request a follow-up question from the backend
  const sendMessage = async () => {
    if (!input.trim()) return;

    const doctorMessage = { text: input, sender: "doctor" };
    setMessages((prev) => [...prev, doctorMessage]);
    updateHistory("Doctor", input);

    try {
      const response = await fetch("http://localhost:8000/ai_followup_doc/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_history: conversationHistory + `\nDoctor: ${input}` }),
      });
      const data = await response.json();
      const botMessage = { text: data.follow_up_question, sender: "bot" };
      setMessages((prev) => [...prev, botMessage]);
      updateHistory("Bot", data.follow_up_question);
    } catch (error) {
      console.error("Error sending message:", error);
    }
    setInput("");
  };

  // Handle file uploads (for MRI image detection)
  const handleFileChange = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setMessages((prev) => [...prev, { text: `📎 ${selectedFile.name}`, sender: "doctor" }]);
    updateHistory("Doctor", `Uploaded file: ${selectedFile.name}`);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/detect_tumor/", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("Tumor Detection API Response:", data);

      let botMessage;
      if (data.tumor_detected && data.tumors.length > 0) {
        // Construct a detailed tumor information message
        const tumorsInfo = data.tumors
          .map(
            (tumor, index) =>
              `🧠 Tumor ${index + 1}\nType: ${tumor.type}\nSize: ${tumor.size}\nLocation: ${tumor.location}\nConfidence: ${tumor.confidence}`
          )
          .join("\n\n");
        botMessage = { text: tumorsInfo, sender: "bot" };
      } else {
        botMessage = { text: "No tumor detected.", sender: "bot" };
      }
      setMessages((prev) => [...prev, botMessage]);
      updateHistory("Bot", botMessage.text);
    } catch (error) {
      console.error("Error uploading file:", error);
      const errorMessage = { text: "⚠️ Error processing the image.", sender: "bot" };
      setMessages((prev) => [...prev, errorMessage]);
      updateHistory("Bot", errorMessage.text);
    }
  };

  // Finalize consultation and request the final diagnosis
  const diagnose = async () => {
    setMessages((prev) => [
      ...prev,
      { text: "Finalizing consultation and generating diagnosis...", sender: "bot" },
    ]);
    updateHistory("Bot", "Finalizing consultation and generating diagnosis...");

    try {
      const response = await fetch("http://localhost:8000/ai_diagnosis_doc/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_history: conversationHistory }),
      });
      const data = await response.json();
      const diagnosisMessage = { text: data.diagnosis, sender: "bot" };
      setMessages((prev) => [...prev, diagnosisMessage]);
      updateHistory("Bot", data.diagnosis);
    } catch (error) {
      console.error("Error diagnosing:", error);
    }
  };

  return (
    <div className={`chatbot-container ${isOpen ? "open" : ""}`}>
      {isOpen ? (
        <div className="chatbot-box">
          <div className="chatbot-header">
            <span>AI Medical Doctor Assistant</span>
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
              placeholder="Enter your observations on the MRI (comma-separated)..."
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

export default DoctorChatbot;
