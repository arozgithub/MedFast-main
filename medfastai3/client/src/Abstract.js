import React from 'react';
import { Link } from 'react-router-dom';
import './Abstract.css';

const Abstract = () => {
    return (
        <div className="abstract-page">
            <div className="container">
                <h1>Abstract</h1>
                <p>
                    A brain tumor is a mass of abnormal cells in the brain. The skull is rigid, and any growth in such a 
                    confined space can create pressure, leading to complications. Brain tumors can be benign (noncancerous) 
                    or malignant (cancerous). As these tumors grow, they may increase intracranial pressure, potentially 
                    causing brain damage and life-threatening conditions.
                </p>
                <Link to="/" className="button">Back to Home</Link>
            </div>
            <footer className="footer">
                <p>&copy; 2025 MedFast AI. All rights reserved.</p>
            </footer>
        </div>
    );
};

export default Abstract;
