# Weapon Detection System (Updated)

A real-time weapon detection application that uses computer vision (YOLO) to identify weapons from camera feeds. 

## Features
- **Real-Time Detection:** Powered by YOLO models for fast and accurate inference.
- **Multiple Camera Support:** Capable of handling feeds from multiple cameras simultaneously.
- **Phone QR Tracking:** Easily connect your mobile phone as a camera source using QR tracking.
- **Cross-Platform Fixes:** Includes specific UI instructions and fixes for Android Chrome camera access.
- **Dual Architecture:** 
  - A robust backend to process frames and run inference.
  - An interactive frontend (Streamlit) for easy viewing and management.

## Project Structure
- `app_build/backend/`: Contains the backend API, YOLO model configurations, and inference logic (`main.py`).
- `app_build/frontend/`: Contains the user interface and camera handling logic (`app.py`).
- `production_artifacts/`: Contains production-ready artifacts like QA reports and Technical Specifications.

## Setup Instructions

### 1. Backend Setup
Navigate to the backend directory and install the requirements:
```bash
cd app_build/backend
pip install -r requirements.txt
```
Run the backend server:
```bash
uvicorn main:app --reload
```

### 2. Frontend Setup
Open a new terminal, navigate to the frontend directory, and install requirements:
```bash
cd app_build/frontend
pip install -r requirements.txt
```
Run the frontend application:
```bash
streamlit run app.py
```

## Troubleshooting
- **Android Chrome Camera Issue:** If you are having trouble accessing your camera on Android via Google Chrome, follow the on-screen instructions in the UI to grant the necessary permissions or apply the fix.
- **Large Model Files:** Make sure any large model files (like `.pt` files over 100MB) are ignored via `.gitignore` if you plan to push changes to GitHub.

## License
MIT License
