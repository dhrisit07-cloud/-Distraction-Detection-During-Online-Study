@echo off
echo ================================================
echo  Focus Guardian - Setup Script
echo ================================================
echo.

echo [1/4] Creating virtual environment with Python 3.11...
py -3.11 -m venv focusenv
if errorlevel 1 (
    echo ERROR: Python 3.11 not found. Please install it from python.org
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call focusenv\Scripts\activate.bat

echo [3/4] Installing required Python packages...
pip install -r requirements.txt

echo [4/4] Installing Next.js frontend packages...
cd frontend
call npm install
cd ..

echo.
echo ================================================
echo  Setup complete! 
echo.
echo  To run the app locally, open TWO terminals:
echo.
echo  Terminal 1 (Backend):
echo    call focusenv\Scripts\activate
echo    uvicorn api.index:app --reload --port 8000
echo.
echo  Terminal 2 (Frontend):
echo    cd frontend
echo    npm run dev
echo ================================================
pause
