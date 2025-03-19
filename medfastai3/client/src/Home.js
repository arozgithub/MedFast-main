import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

const Home = () => {
    return (
        <div className="home">
            <div className="container">
                <div className="content">
                    <h1>Brain Tumor Detection</h1>
                    <h3>Powered by Artificial Intelligence</h3>
                    <p>Fast, accurate, and AI-driven medical analysis for brain tumor detection.</p>
                    <Link to="/Dashboard" className="button">Get Started</Link>
                </div>
            </div>
            <footer className="footer">
                <p>© 2025 | MedFast | All rights reserved.</p>
            </footer>
        </div>
    );
};

export default Home;
