# vercel.py - Configuration for Vercel Python serverless functions
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Optimize for Vercel environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging
os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure output is flushed immediately

# Function to redirect imports if necessary
def redirect_imports():
    """
    Redirects certain imports to more efficient alternatives
    when running in the Vercel environment
    """
    # Example: Use CPU-only versions of libraries when possible
    sys.modules["tensorflow"] = __import__("tensorflow")
    
    # Set Vercel-specific configurations
    os.environ["VERCEL_DEPLOYMENT"] = "1"

# Call the function to set up the environment
redirect_imports()