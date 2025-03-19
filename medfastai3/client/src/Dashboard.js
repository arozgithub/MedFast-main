import React, { useState } from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import Sidebar from './components/Sidebar';
import './Dashboard.css';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend);

const Dashboard = () => {
    // Sample user data
    const userData = [
        { name: 'John Doe', age: 30, condition: 'Healthy' },
        { name: 'Jane Smith', age: 45, condition: 'Diabetes' },
        { name: 'Sam Johnson', age: 60, condition: 'Hypertension' },
    ];

    // Count occurrences of each condition
    const conditionCounts = userData.reduce((acc, user) => {
        acc[user.condition] = (acc[user.condition] || 0) + 1;
        return acc;
    }, {});

    // Chart Data
    const chartData = {
        labels: Object.keys(conditionCounts),
        datasets: [
            {
                label: 'Patient Conditions',
                data: Object.values(conditionCounts),
                backgroundColor: [
                    'rgba(75, 192, 192, 0.6)',  // Healthy
                    'rgba(255, 99, 132, 0.6)',  // Diabetes
                    'rgba(255, 206, 86, 0.6)',  // Hypertension
                ],
            },
        ],
    };

    // Sidebar Toggle
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);

    return (
        <div className="dashboard-container">
            {/* Sidebar */}
            <Sidebar isOpen={isSidebarOpen} />

            {/* Sidebar Toggle Button */}
            <button 
                onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
                className={`sidebar-toggle ${isSidebarOpen ? 'open' : 'closed'}`}
            >
                {isSidebarOpen ? '◄' : '►'}
            </button>

            {/* Main Content */}
            <div className={`main-content ${isSidebarOpen ? 'shifted' : ''}`}>
                <h1>Dashboard</h1>
                <p>Welcome to the MedFast AI Dashboard!</p>

                {/* User Data Section */}
                <div className="user-data">
                    <h2>User Data</h2>
                    <ul>
                        {userData.map((user, index) => (
                            <li key={index}>
                                <strong>{user.name}</strong> - Age: {user.age}, Condition: {user.condition}
                            </li>
                        ))}
                    </ul>
                </div>

                {/* Pie Chart */}
                <div className="chart-container">
                    <h2>Patient Conditions Overview</h2>
                    <Pie data={chartData} />
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
