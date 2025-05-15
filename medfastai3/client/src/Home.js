import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

const Home = () => {
    return (
        <div className="homepage-container">
            <div className="hero-section">
                <h1>Brain Tumor Detection</h1>
                <h3>Powered by Artificial Intelligence</h3>
                <p>Fast, accurate, and AI-driven medical analysis for brain tumor detection.</p>
                <Link to="/Dashboard" className="cta-button">Get Started</Link>
            </div>
            
            <div className="features-section">
                <div className="features-grid">
                    <div className="feature-card">
                        <div className="feature-icon">🧠</div>
                        <h3>Brain Tumor Analysis</h3>
                        <p>Advanced AI algorithms analyze MRI scans to detect and classify brain tumors with high accuracy.</p>
                    </div>
                    
                    <div className="feature-card">
                        <div className="feature-icon">📊</div>
                        <h3>Comprehensive Dashboard</h3>
                        <p>Monitor health metrics and medical analysis with our intuitive dashboard interface.</p>
                    </div>
                    
                    <div className="feature-card">
                        <div className="feature-icon">💬</div>
                        <h3>AI Medical Assistant</h3>
                        <p>Get evidence-based medical insights through our conversational AI chatbot.</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Home;
