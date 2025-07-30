#!/bin/bash

# AnantaAI Startup Script

echo "🚀 Starting AnantaAI - IISc M.Mgt QA System"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if models need to be downloaded
echo "🤖 Checking AI models..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Start the application
echo "🌟 Starting Streamlit application..."
echo "📍 Access the app at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop"

streamlit run app.py
