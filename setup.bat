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

echo [3/4] Installing required packages...
pip install mediapipe==0.10.11 opencv-python ultralytics streamlit numpy

echo [4/4] Verifying installation...
python -c "import cv2; import mediapipe as mp; print('cv2:', cv2.__version__); print('mediapipe:', mp.__version__); print('All packages OK!')"

echo.
echo ================================================
echo  Setup complete! Run the app with:
echo  focusenv\Scripts\activate
echo  streamlit run ui\app.py
echo ================================================
pause
