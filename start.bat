@echo off
REM AnantaAI Full Stack Startup Script for Windows

echo 🚀 Starting AnantaAI - IISc M.Mgt QA System (Full Stack)

REM Check if virtual environment exists and activate it
if exist "venv" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo ⚠️  Virtual environment not found!
    echo    Please run: python setup_venv.py
    pause
    exit /b 1
)

REM Step 1: Start backend in background
echo 🔧 Starting backend server...
start "AnantaAI Backend" cmd /k "cd backend && python main.py"

REM Give backend time to start
timeout /t 3 /nobreak >nul

REM Step 2: Navigate to frontend directory
echo 📱 Setting up frontend...
cd frontend

REM Step 3: Install npm dependencies
echo 📥 Installing npm dependencies...
call npm install

REM Step 4: Install additional Tailwind packages
echo 🎨 Installing Tailwind CSS packages...
call npm install @tailwindcss/forms @tailwindcss/typography @tailwindcss/aspect-ratio --save-dev

REM Step 5: Start frontend development server
echo 🌟 Starting frontend development server...
echo.
echo 🎯 Services starting:
echo    📡 Backend API: http://localhost:8000
echo    🖥️  Frontend: http://localhost:5173
echo.
echo 🛑 Close this window or press Ctrl+C to stop frontend
echo 🛑 Close the backend window to stop the API server
echo.

REM Start frontend (this will run in foreground)
call npm run dev

echo.
echo 👋 Frontend stopped. Backend may still be running in separate window.
pause
