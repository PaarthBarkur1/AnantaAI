@echo off
REM AnantaAI Full Stack Startup Script for Windows

REM Ensure the console supports UTF-8 for emojis and special characters
chcp 65001 > nul

REM Enable delayed expansion for robust variable handling, especially with complex command strings
setlocal enabledelayedexpansion

echo 🚀 Starting AnantaAI - IISc M.Mgt QA System (Full Stack)

REM Ensure the script always runs from its own directory (the project root)
pushd "%~dp0"
echo Current working directory: %cd%

REM --- Virtual Environment Activation ---
echo.
echo 🔧 Checking for and activating virtual environment...

REM Prioritize .venv (common for projects, matches your tracebacks)
if exist ".venv" (
    echo    Activating .venv...
    call .venv\Scripts\activate
) else (
    REM If .venv not found, check for venv (without the dot)
    if exist "venv" (
        echo    Activating venv...
        call venv\Scripts\activate
    ) else (
        REM If neither is found, exit with an error
        echo ⚠️ Virtual environment not found!
        echo    Please run: python setup_venv.py
        pause
        exit /b 1
    )
)

REM --- Determine Python Executable Path ---
REM After activation, the PATH should be updated, but for robust explicit calls (like with 'start'),
REM it's safer to determine the absolute path to the venv's python.exe.
set "PYTHON_EXE=!cd!\.venv\Scripts\python.exe"
if not exist "!PYTHON_EXE!" (
    set "PYTHON_EXE=!cd!\venv\Scripts\python.exe"
)

REM Final verification that a Python executable was found in the expected venv location
if not exist "!PYTHON_EXE!" (
    echo ERROR: venv Python executable not found at "!PYTHON_EXE!" after activation attempt.
    echo Please ensure your virtual environment is correctly set up.
    pause
    exit /b 1
)

echo    Python executable detected: "!PYTHON_EXE!"
echo.

REM --- Step 1: Start Backend Server in a New Window ---
echo 🔧 Starting backend server (API will be at http://localhost:8000)...
REM Construct the full command string for the backend server.
REM Note: ^> and 2^>^&1 are used to correctly redirect output to backend_log.txt in batch.
set "BACKEND_COMMAND=!PYTHON_EXE! -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level debug --reload"

REM Use 'start' to open a new command prompt window.
REM /k keeps the new window open after the command runs.
start "AnantaAI Backend" cmd /k "!BACKEND_COMMAND!"

REM Give the backend a few seconds to initialize
echo    Giving backend a moment to start...
timeout /t 5 /nobreak > nul

REM --- Step 2: Navigate to Frontend Directory ---
echo 📱 Preparing frontend...
if not exist "frontend" (
    echo ERROR: 'frontend' directory not found. Please ensure project structure is correct.
    pause
    exit /b 1
)
cd frontend
echo    Changed directory to frontend: %cd%

REM --- Step 3: Install npm Dependencies (if not already installed) ---
echo 📥 Installing npm dependencies (if needed)...
call npm install
if errorlevel 1 (
    echo ERROR: npm install failed. Please check your Node.js and npm setup.
    pause
    exit /b 1
)

REM --- Step 4: Install Additional Tailwind Packages (if not already installed) ---
echo 🎨 Installing Tailwind CSS packages (if needed)...
call npm install @tailwindcss/forms @tailwindcss/typography @tailwindcss/aspect-ratio --save-dev
if errorlevel 1 (
    echo WARNING: Tailwind package installation failed. Frontend might not display correctly.
)

REM --- Step 5: Start Frontend Development Server (in current window) ---
echo.
echo 🌟 Starting frontend development server (http://localhost:5173)...
echo.
echo 🎯 Services expected to be running:
echo    📡 Backend API: http://localhost:8000
echo    🖥️  Frontend: http://localhost:5173
echo.
echo 🛑 To stop the frontend, close this window or press Ctrl+C here.
echo 🛑 To stop the backend API, close the separate "AnantaAI Backend" window.
echo.

REM 'call npm run dev' will keep this window occupied until the frontend server is stopped
call npm run dev

echo.
echo 👋 Frontend development server stopped.
echo    Remember to close the "AnantaAI Backend" window to stop the API server.
pause

REM --- Script Cleanup ---
REM Restore original environment variables and directory
endlocal