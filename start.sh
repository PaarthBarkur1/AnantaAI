#!/bin/bash

# AnantaAI Full Stack Startup Script

echo "🚀 Starting AnantaAI - IISc M.Mgt QA System (Full Stack)"

# Function to cleanup background processes on exit
cleanup() {
    echo "🛑 Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Step 1: Go to parent folder
echo "� Moving to parent directory..."
cd ..

# Step 2: Start backend in background
echo "🔧 Starting backend server..."
python -m AnantaAI.backend.main &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Step 3: Navigate to frontend directory
echo "� Setting up frontend..."
cd AnantaAI/frontend

# Step 4: Install npm dependencies
echo "📥 Installing npm dependencies..."
npm install

# Step 5: Install additional Tailwind packages
echo "🎨 Installing Tailwind CSS packages..."
npm install @tailwindcss/forms @tailwindcss/typography @tailwindcss/aspect-ratio --save-dev

# Step 6: Start frontend development server
echo "🌟 Starting frontend development server..."
echo ""
echo "🎯 Services starting:"
echo "   � Backend API: http://localhost:8000"
echo "   🖥️  Frontend: http://localhost:5173"
echo ""
echo "🛑 Press Ctrl+C to stop all services"
echo ""

# Start frontend (this will run in foreground)
npm run dev

# If we reach here, frontend has stopped, so cleanup
cleanup
