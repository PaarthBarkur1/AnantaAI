@echo off
REM AnantaAI Startup Script for Windows

echo 🚀 Starting AnantaAI - IISc M.Mgt QA System

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

REM Install/update dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Check if models need to be downloaded
echo 🤖 Checking AI models...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

REM Start the application
echo 🌟 Starting Streamlit application...
echo 📍 Access the app at: http://localhost:8501
echo 🛑 Press Ctrl+C to stop

streamlit run app.py
