@echo off
echo Starting Festus Bombo Data Science Portfolio...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies if requirements file exists
if exist "requirements_local.txt" (
    echo Installing dependencies...
    pip install -r requirements_local.txt
    echo.
) else (
    echo Installing core dependencies...
    pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn scipy
    echo.
)

REM Run the Streamlit app
echo Starting portfolio...
echo Portfolio will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
streamlit run app.py

pause