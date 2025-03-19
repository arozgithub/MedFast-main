import React from 'react';
import { Link } from 'react-router-dom';
import './Contact.css';

const Contact = () => {
    return (
        <div className="contact-page">
            <div className="contact-container">
                <h1>Get In Touch</h1>
                <p>Our inbox is always open. We'll try our best to get back to you!</p>
                <hr />

                <form>
                    <input type="text" name="name" placeholder="Your Name" required />
                    <input type="email" name="email" placeholder="Your Email" required />
                    <textarea name="message" placeholder="Your Message" required></textarea>
                    
                    <button type="submit" className="send-btn">Send Message</button>
                </form>

                <p className="back-link">
                    <Link to="/">Go Back Home</Link>
                </p>
            </div>

            <footer className="footer">
                <p>&copy; 2025 MedFast AI. All rights reserved.</p>
            </footer>
        </div>
    );
};

export default Contact;
