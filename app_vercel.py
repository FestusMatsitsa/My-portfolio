from flask import Flask
import streamlit.web.cli as stcli
import sys
import os

app = Flask(__name__)

@app.route('/')
def main():
    # Set up Streamlit configuration for Vercel
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    # Run Streamlit app
    sys.argv = ["streamlit", "run", "app.py", "--server.port", "8080"]
    stcli.main()

if __name__ == '__main__':
    main()