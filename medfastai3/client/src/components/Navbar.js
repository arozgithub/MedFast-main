import React, { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import "./Navbar.css"; // Import CSS file
import DarkModeToggle from "./DarkModeToggle";

const Navbar = () => {
  const [darkMode, setDarkMode] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.body.classList.toggle("dark-mode");
  };

  const handleAboutClick = (e) => {
    e.preventDefault(); // Prevent default link behavior

    if (location.pathname === "/") {
      // Already on home page, just scroll to About section
      document.getElementById("about")?.scrollIntoView({ behavior: "smooth" });
    } else {
      // Navigate to home page first, then scroll after rendering
      navigate("/");

      setTimeout(() => {
        document.getElementById("about")?.scrollIntoView({ behavior: "smooth" });
      }, 300);
    }
  };

  return (
    <nav className={`navbar ${darkMode ? "dark" : ""}`}>
      <div className="logo">MEDFAST</div>
      <ul className="nav-links">
      <li><Link to="/" className="active">Home</Link></li>
                <li><Link to="/abstract">Abstract</Link></li>
                <li><Link to="/login">Login</Link></li>
                <li><Link to="/signup">Register</Link></li>
                <li><Link to="/contact">Contact</Link></li>
                <li><Link to="/dashboard">Dashboard</Link></li>
                <li><Link to="/doctor">Doctor</Link></li>
      </ul>

      <DarkModeToggle />
    </nav>
  );
};

export default Navbar;
