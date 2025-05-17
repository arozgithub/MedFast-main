// API configuration file for the MedFast application
// This file helps manage API endpoints between frontend and backend

// Determine the base URL for API requests based on environment
const getBaseUrl = () => {
  // Check for Render-specific environment variable first
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  // In production (Vercel), API calls go to the same domain but with /api prefix
  else if (process.env.NODE_ENV === 'production') {
    return '/api';
  }
  // In development, use localhost with the backend port
  return 'http://localhost:8000';
};

const API_BASE_URL = getBaseUrl();

// Export API endpoints for use throughout the application
export const API_ENDPOINTS = {
  detectTumor: `${API_BASE_URL}/detect_tumor/`,
  detectPneumonia: `${API_BASE_URL}/detect_pneumonia/`,
  aiDiagnosis: `${API_BASE_URL}/ai_diagnosis/`,
  aiFollowup: `${API_BASE_URL}/ai_followup/`,
  aiDiagnosisDoc: `${API_BASE_URL}/ai_diagnosis_doc/`,
  aiFollowupDoc: `${API_BASE_URL}/ai_followup_doc/`,
  detectTumorH5: `${API_BASE_URL}/detect_tumor_h5/`,
  predictDiabetes: `${API_BASE_URL}/predict_diabetes/`
};

export default API_ENDPOINTS;