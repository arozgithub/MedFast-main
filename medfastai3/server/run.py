import uvicorn
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    print("Starting MedFast AI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 