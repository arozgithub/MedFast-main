import React, { useState } from "react";
import emailjs from "emailjs-com";
import { Link } from 'react-router-dom';
import "./Contact.css";

const Contact = () => {
  const [formData, setFormData] = useState({ name: "", email: "", message: "" });
  const [submitted, setSubmitted] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    emailjs
      .send(
        "service_iox3nii",
        "template_e90enub",
        formData,
        "R0tQ8FHp27R1yrKVX"
      )
      .then(
        (response) => {
          console.log("SUCCESS!", response.status, response.text);
          setSubmitted(true);
          setFormData({ name: "", email: "", message: "" });
          // Hide the success message after 3 seconds
          setTimeout(() => setSubmitted(false), 3000);
        },
        (err) => {
          console.error("FAILED...", err);
        }
      );
  };

  return (
    <div className="contact-container">
      <div className="contact-content">
        <div className="contact-header">
          <h1>Get In Touch</h1>
          <p>Our inbox is always open. We'll try our best to get back to you!</p>
        </div>

        {submitted && (
          <div className="submit-message">
            ✅ Message sent successfully!
          </div>
        )}

        <div className="contact-form">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="name">Name</label>
              <input
                type="text"
                id="name"
                name="name"
                placeholder="Your Name"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                name="email"
                placeholder="Your Email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="message">Message</label>
              <textarea
                id="message"
                name="message"
                placeholder="Your Message"
                value={formData.message}
                onChange={handleChange}
                required
              ></textarea>
            </div>
            
            <button type="submit" className="submit-button">
              🚀 Send Message
            </button>
          </form>
        </div>

        <div className="back-link">
          <Link to="/">Go Back Home</Link>
        </div>
      </div>
      
      {/* Trigger area for footer hover */}
      <div className="footer-trigger"></div>
      
      {/* Footer with contact information */}
      <footer className="contact-footer">
        <div className="footer-container">
          <h3>Other Ways to Reach Us</h3>
          <div className="footer-info-grid">
            <div className="footer-info-item">
              <h4>Email</h4>
              <p>support@medfastai.com</p>
            </div>
            <div className="footer-info-item">
              <h4>Phone</h4>
              <p>+1 (800) 555-0123</p>
            </div>
            <div className="footer-info-item">
              <h4>Location</h4>
              <p>123 Medical Center Ave, Health City, HC 12345</p>
            </div>
          </div>
          <div className="footer-home-link">
            <Link to="/">Go Back Home</Link>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Contact;
