import React, { useState, useEffect, useRef } from "react";
import { Send, X, MessageCircle, Plus, Activity, Mic, Volume2, VolumeX } from "lucide-react";
import { jsPDF } from "jspdf";
import "./chatbot.css";

const Chatbot = () => {
  const [messages, setMessages] = useState([
    { text: "Hi! How can I assist with your diagnosis?", sender: "bot" },
  ]);
  const [conversationHistory, setConversationHistory] = useState("");
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  
  // Create a ref to the input element so we can focus it
  const inputRef = useRef(null);
  const recognitionRef = useRef(null);
  const speechSynthesisRef = useRef(window.speechSynthesis);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
    
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      
      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInput(transcript);
        setTimeout(() => {
          sendMessage(transcript);
        }, 500);
      };
      
      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        setIsListening(false);
      };
    }
    
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
      if (speechSynthesisRef.current) {
        speechSynthesisRef.current.cancel();
      }
    };
  }, [isOpen]);

  const updateHistory = (sender, text) => {
    const updatedHistory = conversationHistory + `\n${sender}: ${text}`;
    setConversationHistory(updatedHistory);
  };

  const sendMessage = async (voiceInput = null) => {
    const messageText = voiceInput || input;
    if (!messageText.trim()) return;
    
    const userMessage = { text: messageText, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    updateHistory("User", messageText);

    try {
      const response = await fetch("http://localhost:8000/ai_followup/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_history: conversationHistory + `\nUser: ${messageText}`,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      const responseText = data.follow_up_question || "I'm not sure how to respond to that.";
      const botMessage = { text: responseText, sender: "bot" };
      setMessages((prev) => [...prev, botMessage]);
      updateHistory("Bot", responseText);
      
      if (voiceEnabled) {
        speakText(responseText);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = { text: "Sorry, I'm having trouble connecting to the server right now.", sender: "bot" };
      setMessages((prev) => [...prev, errorMessage]);
      updateHistory("Bot", errorMessage.text);
      
      if (voiceEnabled) {
        speakText(errorMessage.text);
      }
    }
    setInput("");
  };

  const startListening = () => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (error) {
        console.error("Error starting speech recognition:", error);
      }
    } else {
      alert("Speech recognition is not supported in your browser.");
    }
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  const toggleVoice = () => {
    setVoiceEnabled(!voiceEnabled);
    if (isSpeaking && !voiceEnabled) {
      speechSynthesisRef.current.cancel();
      setIsSpeaking(false);
    }
  };

  const speakText = (text) => {
    if (!voiceEnabled) return;
    
    speechSynthesisRef.current.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    
    const voices = speechSynthesisRef.current.getVoices();
    const preferredVoice = voices.find(voice => 
      voice.name.includes("Female") || voice.name.includes("Google") || voice.lang === 'en-US'
    );
    
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }
    
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = (event) => {
      console.error("Speech synthesis error:", event);
      setIsSpeaking(false);
    };
    
    speechSynthesisRef.current.speak(utterance);
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
      
      if (voiceEnabled) {
        speakText(botMessage.text);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      const errorMessage = { text: "⚠️ Error processing the image.", sender: "bot" };
      setMessages((prev) => [...prev, errorMessage]);
      updateHistory("Bot", errorMessage.text);
    }
  };

  // Function to generate a PDF report from the conversation history
  const generatePDFReport = () => {
    const doc = new jsPDF();
    const lines = conversationHistory.split("\n");
    let y = 10;
    doc.setFontSize(12);
    lines.forEach((line) => {
      doc.text(line, 10, y);
      y += 7;
      if (y > 280) {
        doc.addPage();
        y = 10;
      }
    });
    doc.save("chat_report.pdf");
  };

  const diagnose = async () => {
    setMessages((prev) => [
      ...prev,
      { text: "Diagnosing your condition...", sender: "bot" },
    ]);
    updateHistory("Bot", "Diagnosing your condition...");
    try {
      const response = await fetch("http://localhost:8000/ai_diagnosis/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_history: conversationHistory }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      const diagnosisText = data.diagnosis || "I couldn't generate a diagnosis at this time.";
      const diagnosisMessage = { text: diagnosisText, sender: "bot" };
      setMessages((prev) => [...prev, diagnosisMessage]);
      updateHistory("Bot", diagnosisText);
      
      if (voiceEnabled) {
        speakText(diagnosisText);
      }
    } catch (error) {
      console.error("Error diagnosing:", error);
      const errorMessage = { text: "Sorry, I'm having trouble connecting to the diagnosis service.", sender: "bot" };
      setMessages((prev) => [...prev, errorMessage]);
      updateHistory("Bot", errorMessage.text);
      
      if (voiceEnabled) {
        speakText(errorMessage.text);
      }
    }
  };

  // Handler for keypress events in the input field
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className={`chatbot-container ${isOpen ? "open" : ""}`}>
      {isOpen ? (
        <div className="chatbot-box">
          <div className="chatbot-header text-center">
            <span className="text-red-400">AI Medical Assistant</span>
            <div className="voice-controls">
              <button 
                onClick={toggleVoice} 
                title={voiceEnabled ? "Disable voice" : "Enable voice"} 
                className="voice-toggle"
              >
                {voiceEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
              </button>
            </div>
            <X className="close-btn" onClick={() => setIsOpen(false)} />
          </div>
          <div className="chatbot-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.sender}`}>
                {msg.text}
              </div>
            ))}
          </div>
          <div className="chatbot-actions">
            <div className="chatbot-input">
              <input
                type="text"
                placeholder={isListening ? "Listening..." : "Describe your symptoms..."}
                value={input}
                ref={inputRef}
                onKeyDown={handleKeyDown}
                onChange={(e) => setInput(e.target.value)}
                disabled={isListening}
              />
              <div className="button-container">
                <button onClick={() => sendMessage()} disabled={isListening || !input.trim()} className="send-btn">
                  <Send size={18} />
                </button>
                <button 
                  onClick={isListening ? stopListening : startListening}
                  className={`mic-btn ${isListening ? 'active' : ''}`}
                  title={isListening ? "Stop listening" : "Start voice input"}
                >
                  <Mic size={18} />
                </button>
                <label className="upload-btn">
                  <Plus size={18} />
                  <input
                    type="file"
                    onChange={handleFileChange}
                    style={{ display: "none" }}
                  />
                </label>
                <button className="diagnose-btn" onClick={diagnose} title="Get diagnosis">
                  <Activity size={18} />
                </button>
              </div>
              <div className="pdf-report">
                <button onClick={generatePDFReport} className="pdf-button">
                  Generate Report PDF
                </button>
              </div>
            </div>
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
