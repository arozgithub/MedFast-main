import React, { useState, useEffect } from "react";
import emoji from "react-easy-emoji";
import "./DarkModeToggle.css";

const DarkModeToggle = () => {
  const [darkMode, setDarkMode] = useState(localStorage.getItem("theme") === "dark");

  useEffect(() => {
    if (darkMode) {
      document.body.classList.add("dark");  // Add dark class to the body
      localStorage.setItem("theme", "dark");
    } else {
      document.body.classList.remove("dark");  // Remove dark class from the body
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  return (
    <div className="fixed top-4 right-4 flex items-center space-x-2 bg-gray-800 p-2 rounded-lg shadow-lg">
      <span className="text-white">{darkMode ? emoji("🌜") : emoji("☀️")}</span>
      <span className="text-white">{darkMode ? "Dark Mode" : "Light Mode"}</span>
      <label className="switch">
        <input type="checkbox" checked={darkMode} onChange={() => setDarkMode(!darkMode)} />
        <span className="slider round"></span>
      </label>
    </div>
  );
};

export default DarkModeToggle;
