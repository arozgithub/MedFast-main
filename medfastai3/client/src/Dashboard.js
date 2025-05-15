import React, { useState, useEffect } from 'react';
import { Pie, Bar, Line } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title } from 'chart.js';
import Sidebar from './components/Sidebar';
import DarkModeToggle from './components/DarkModeToggle';
import './Dashboard.css';
import { Link } from 'react-router-dom';

// Register all the ChartJS components
ChartJS.register(
    ArcElement, 
    Tooltip, 
    Legend, 
    CategoryScale, 
    LinearScale, 
    BarElement, 
    PointElement, 
    LineElement, 
    Title
);

const Dashboard = () => {
    // State to track dark mode
    const [isDarkMode, setIsDarkMode] = useState(localStorage.getItem("theme") === "dark");
    
    // Monitor changes to the body's dark class to update our state
    useEffect(() => {
        const checkDarkMode = () => {
            setIsDarkMode(document.body.classList.contains('dark'));
        };
        
        // Check initial state
        checkDarkMode();
        
        // Setup a mutation observer to detect changes to body classes
        const observer = new MutationObserver(checkDarkMode);
        observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
        
        return () => observer.disconnect();
    }, []);
    
    // Sample patient data with more medical details
    const patientData = [
        { id: 1, name: 'John Doe', age: 30, condition: 'Healthy', lastVisit: '2025-05-01', analysis: { tumor: false, diabetes: false, pneumonia: false } },
        { id: 2, name: 'Jane Smith', age: 45, condition: 'Diabetes', lastVisit: '2025-05-05', analysis: { tumor: false, diabetes: true, pneumonia: false } },
        { id: 3, name: 'Sam Johnson', age: 60, condition: 'Hypertension', lastVisit: '2025-04-28', analysis: { tumor: false, diabetes: false, pneumonia: false } },
        { id: 4, name: 'Emma Wilson', age: 28, condition: 'Pneumonia', lastVisit: '2025-05-10', analysis: { tumor: false, diabetes: false, pneumonia: true } },
        { id: 5, name: 'Robert Brown', age: 52, condition: 'Brain Tumor', lastVisit: '2025-05-12', analysis: { tumor: true, diabetes: false, pneumonia: false } },
    ];

    // Count occurrences of each condition
    const conditionCounts = patientData.reduce((acc, patient) => {
        acc[patient.condition] = (acc[patient.condition] || 0) + 1;
        return acc;
    }, {});

    // Chart options with theme-based configuration
    const getChartOptions = (title) => ({
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { 
                position: 'top',
                labels: { 
                    color: isDarkMode ? '#e0e0e0' : '#333' 
                }
            },
            title: { 
                display: true, 
                text: title,
                color: isDarkMode ? '#e0e0e0' : '#2c3e50'
            }
        },
        scales: {
            y: {
                ticks: {
                    color: isDarkMode ? '#e0e0e0' : '#333'
                },
                grid: {
                    color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
                }
            },
            x: {
                ticks: {
                    color: isDarkMode ? '#e0e0e0' : '#333'
                },
                grid: {
                    color: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
                }
            }
        }
    });

    // Pie Chart Data for Conditions
    const pieChartData = {
        labels: Object.keys(conditionCounts),
        datasets: [
            {
                label: 'Patient Conditions',
                data: Object.values(conditionCounts),
                backgroundColor: [
                    'rgba(75, 192, 192, 0.7)',  // Healthy
                    'rgba(255, 99, 132, 0.7)',  // Diabetes
                    'rgba(255, 206, 86, 0.7)',  // Hypertension
                    'rgba(54, 162, 235, 0.7)',  // Pneumonia
                    'rgba(153, 102, 255, 0.7)', // Brain Tumor
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(153, 102, 255, 1)',
                ],
                borderWidth: 1,
            },
        ],
    };

    // Mock data for AI detection statistics
    const aiDetectionData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        datasets: [
            {
                label: 'Tumor Detection Rate',
                data: [85, 87, 90, 93, 95],
                backgroundColor: 'rgba(153, 102, 255, 0.7)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1,
            },
            {
                label: 'Diabetes Prediction Accuracy',
                data: [90, 92, 91, 93, 96],
                backgroundColor: 'rgba(255, 99, 132, 0.7)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
            },
            {
                label: 'Pneumonia Detection Accuracy',
                data: [82, 85, 88, 90, 94],
                backgroundColor: 'rgba(54, 162, 235, 0.7)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
            }
        ]
    };

    // Mock data for monthly consultations
    const monthlyConsultations = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        datasets: [
            {
                label: 'AI Consultations',
                data: [120, 150, 180, 210, 250],
                backgroundColor: isDarkMode ? 'rgba(75, 192, 192, 0.3)' : 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
            }
        ]
    };

    // Sidebar Toggle
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    
    // Stats for quick cards
    const [stats, setStats] = useState({
        totalPatients: patientData.length,
        activeDiagnoses: patientData.filter(p => p.condition !== 'Healthy').length,
        aiAccuracy: 94, // Mock average AI accuracy percentage
        pendingConsults: 8 // Mock number of pending consultations
    });

    // Current date for dashboard
    const currentDate = new Date().toLocaleDateString('en-US', { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    });

    return (
        <div className={`dashboard-container ${isDarkMode ? 'dark-theme' : 'light-theme'}`}>
            {/* Sidebar */}
            <Sidebar isOpen={isSidebarOpen} />

            {/* Sidebar Toggle Button */}
            <button 
                onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
                className={`sidebar-toggle ${isSidebarOpen ? 'open' : 'closed'}`}
            >
                {isSidebarOpen ? '◄' : '►'}
            </button>

            {/* Dark Mode Toggle Component */}
            {/* <div className="dark-mode-toggle-container">
                <DarkModeToggle />
            </div> */}

            {/* Main Content */}
            <div className={`main-content ${isSidebarOpen ? 'shifted' : ''}`}>
                <div className="dashboard-header">
                    <h1>MedFast AI Dashboard</h1>
                    <p className="date-display">{currentDate}</p>
                </div>

                {/* Stats Cards */}
                <div className="stats-container">
                    <div className="stat-card">
                        <h3>Total Patients</h3>
                        <p className="stat-value">{stats.totalPatients}</p>
                        <i className="stat-icon fas fa-users"></i>
                    </div>
                    <div className="stat-card">
                        <h3>Active Diagnoses</h3>
                        <p className="stat-value">{stats.activeDiagnoses}</p>
                        <i className="stat-icon fas fa-procedures"></i>
                    </div>
                    <div className="stat-card">
                        <h3>AI Accuracy</h3>
                        <p className="stat-value">{stats.aiAccuracy}%</p>
                        <i className="stat-icon fas fa-brain"></i>
                    </div>
                    <div className="stat-card">
                        <h3>Pending Consults</h3>
                        <p className="stat-value">{stats.pendingConsults}</p>
                        <i className="stat-icon fas fa-clock"></i>
                    </div>
                </div>

                {/* Service Quick Links */}
                <div className="services-container">
                    <h2>Medical AI Services</h2>
                    <div className="service-cards">
                        <Link to="/tumor-detection" className="service-card">
                            <h3>Brain Tumor Detection</h3>
                            <p>Upload MRI scans to detect and classify brain tumors using YOLO AI</p>
                        </Link>
                        <Link to="/segmentation" className="service-card">
                            <h3>Tumor Segmentation</h3>
                            <p>Visualize tumor boundaries and get detailed segmentation analysis</p>
                        </Link>
                        <Link to="/xray-analysis" className="service-card">
                            <h3>Pneumonia Detection</h3>
                            <p>Analyze chest X-rays for pneumonia detection</p>
                        </Link>
                        <Link to="/diabetes-detection" className="service-card">
                            <h3>Diabetes Prediction</h3>
                            <p>Predict diabetes risk based on clinical indicators</p>
                        </Link>
                    </div>
                </div>

                <div className="charts-container">
                    {/* First row of charts */}
                    <div className="chart-row">
                        {/* Patient Conditions Pie Chart */}
                        <div className="chart-card">
                            <h2>Patient Conditions</h2>
                            <div className="chart-wrapper">
                                <Pie data={pieChartData} options={getChartOptions('Patient Condition Distribution')} />
                            </div>
                        </div>
                        
                        {/* AI Detection Accuracy Bar Chart */}
                        <div className="chart-card">
                            <h2>AI Detection Metrics</h2>
                            <div className="chart-wrapper">
                                <Bar data={aiDetectionData} options={getChartOptions('AI Detection Accuracy (%)')} />
                            </div>
                        </div>
                    </div>

                    {/* Second row of charts */}
                    <div className="chart-row">
                        {/* Monthly Consultations Line Chart */}
                        <div className="chart-card full-width">
                            <h2>Monthly AI Consultations</h2>
                            <div className="chart-wrapper">
                                <Line data={monthlyConsultations} options={getChartOptions('Monthly AI Consultations')} />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Recent Patients Table */}
                <div className="patients-container">
                    <h2>Recent Patients</h2>
                    <div className="table-container">
                        <table className="patients-table">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Name</th>
                                    <th>Age</th>
                                    <th>Condition</th>
                                    <th>Last Visit</th>
                                    <th>AI Analysis</th>
                                </tr>
                            </thead>
                            <tbody>
                                {patientData.map((patient) => (
                                    <tr key={patient.id}>
                                        <td>{patient.id}</td>
                                        <td>{patient.name}</td>
                                        <td>{patient.age}</td>
                                        <td>
                                            <span className={`condition-badge ${patient.condition.toLowerCase().replace(' ', '-')}`}>
                                                {patient.condition}
                                            </span>
                                        </td>
                                        <td>{patient.lastVisit}</td>
                                        <td>
                                            {patient.analysis.tumor && <span className="analysis-badge tumor">Tumor</span>}
                                            {patient.analysis.diabetes && <span className="analysis-badge diabetes">Diabetes</span>}
                                            {patient.analysis.pneumonia && <span className="analysis-badge pneumonia">Pneumonia</span>}
                                            {!patient.analysis.tumor && !patient.analysis.diabetes && !patient.analysis.pneumonia && 
                                                <span className="analysis-badge healthy">Healthy</span>
                                            }
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
