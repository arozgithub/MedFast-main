import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = ({ isOpen }) => {
    const location = useLocation();  // Get the current route
    const [activeLink, setActiveLink] = useState(location.pathname);  // Initialize active link with current route

    // Function to handle the active link
    const handleLinkClick = (path) => {
        setActiveLink(path);  // Set active link based on clicked path
    };

    return (
        <div className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
            <h3 className="sidebar-title">Menu</h3>
            <ul className="sidebar-list">
                <li>
                    <Link
                        to="/segmentation"
                        className={`sidebar-link ${activeLink === '/segmentation' ? 'active' : ''}`}
                        onClick={() => handleLinkClick('/segmentation')}
                    >
                        Segmentation
                    </Link>
                </li>
                <li>
                    <Link
                        to="/tumor-detection"
                        className={`sidebar-link ${activeLink === '/tumor-detection' ? 'active' : ''}`}
                        onClick={() => handleLinkClick('/tumor-detection')}
                    >
                        Tumor Detection
                    </Link>
                </li>
                <li>
                    <Link
                        to="/xray-analysis"
                        className={`sidebar-link ${activeLink === '/xray-analysis' ? 'active' : ''}`}
                        onClick={() => handleLinkClick('/xray-analysis')}
                    >
                        X-RAY Analysis
                    </Link>
                </li>
                <li>
                    <Link
                        to="/diabetes-detection"
                        className={`sidebar-link ${activeLink === '/diabetes-detection' ? 'active' : ''}`}
                        onClick={() => handleLinkClick('/diabetes-detection')}
                    >
                        Diabetes Detection
                    </Link>
                </li>
            </ul>
        </div>
    );
};

export default Sidebar;
