import React, { useState } from "react";
import { Send, X, MessageCircle, Plus, Activity } from "lucide-react";
import "./chatbot.css";

const Chatbot = () => {
  const [messages, setMessages] = useState([
    { text: "Hi! How can I assist with your diagnosis?", sender: "bot" },
  ]);
  // Maintain conversation history as a concatenated string
  const [conversationHistory, setConversationHistory] = useState("");
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [file, setFile] = useState(null);

  const updateHistory = (sender, text) => {
    // Append new messages to conversationHistory with a newline separator
    const updatedHistory = conversationHistory + `\n${sender}: ${text}`;
    setConversationHistory(updatedHistory);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Update messages and conversation history with user's input
    const userMessage = { text: input, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    updateHistory("User", input);

    try {
      const response = await fetch("http://localhost:8000/ai_followup/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_history: conversationHistory + `\nUser: ${input}` }),
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

  const handleFileChange = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
  
    setFile(selectedFile);
    setMessages((prev) => [...prev, { text: `📎 ${selectedFile.name}`, sender: "user" }]);
    updateHistory("User", `Uploaded file: ${selectedFile.name}`);
  
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
        // Build a message string based on tumor detection results
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

  const diagnose = async () => {
    // Optionally add a placeholder bot message
    setMessages((prev) => [...prev, { text: "Diagnosing your condition...", sender: "bot" }]);
    updateHistory("Bot", "Diagnosing your condition...");

    try {
      const response = await fetch("http://localhost:8000/ai_diagnosis/", {
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
          <div className="chatbot-header text-center">
            <span className="text-red-400">AI Medical Assistant</span>
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
