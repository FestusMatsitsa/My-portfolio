#!/bin/bash

echo "Starting Festus Bombo Data Science Portfolio..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ -f "requirements_local.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements_local.txt
    echo
else
    echo "Installing core dependencies..."
    pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn scipy
    echo
fi

# Run the Streamlit app
echo "Starting portfolio..."
echo "Portfolio will open at: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the server"
echo
streamlit run app.py