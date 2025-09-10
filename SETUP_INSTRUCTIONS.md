# VS Code Setup Instructions for Data Science Portfolio

## Quick Start Guide

### 1. Download and Extract
- Download all files from this portfolio project
- Extract to a folder (e.g., `data-science-portfolio`)

### 2. Open in VS Code
```bash
# Navigate to project folder
cd data-science-portfolio

# Open in VS Code
code .
```

### 3. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements_local.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn scipy
```

### 5. Run the Portfolio
```bash
streamlit run app.py
```

### 6. Access Your Portfolio
- Open browser to: `http://localhost:8501`
- Portfolio will automatically open

## Project Structure
```
data-science-portfolio/
├── app.py                          # Main application
├── pages/
│   ├── 1_Experience.py            # Experience page
│   ├── 2_Skills.py                # Skills page
│   ├── 3_Data_Science_Toolkit.py  # Interactive tools
│   ├── 4_Projects.py              # Project showcase
│   └── 5_Contact_Me.py            # Contact form
├── utils/
│   ├── data_analysis.py           # Analysis utilities
│   └── ml_models.py               # ML utilities
├── attached_assets/
│   └── My Resume.pdf              # Your resume
├── .streamlit/
│   └── config.toml                # App configuration
├── messages/                      # Contact form messages
├── requirements_local.txt         # Dependencies
└── README.md                      # Documentation
```

## VS Code Extensions (Recommended)
- Python
- Pylance
- Python Docstring Generator
- GitLens

## Deployment Options

### Option 1: Streamlit Cloud (Easiest)
1. Upload to GitHub
2. Visit share.streamlit.io
3. Connect repository
4. Deploy automatically

### Option 2: Vercel
1. Upload to GitHub
2. Connect to Vercel
3. Uses included vercel.json config

### Option 3: Heroku
1. Upload to GitHub
2. Connect to Heroku
3. Uses included Procfile

## Troubleshooting

### Common Issues:

**Port Already in Use:**
```bash
# Kill existing Streamlit processes
pkill -f streamlit
# Then restart
streamlit run app.py
```

**Module Not Found:**
```bash
# Ensure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
# Reinstall dependencies
pip install -r requirements_local.txt
```

**Contact Form Messages:**
- Messages are saved in `/messages/` folder
- Each message creates a JSON file with timestamp
- Check this folder for new inquiries

## Customization

### Update Contact Information:
- Edit `pages/5_Contact_Me.py`
- Update email, phone, GitHub links

### Add New Projects:
- Edit `pages/4_Projects.py`
- Add project cards with your details

### Modify Styling:
- Update CSS in each page file
- Change colors in `.streamlit/config.toml`

### Add New Pages:
- Create new file in `/pages/` folder
- Name format: `6_New_Page.py`
- Streamlit automatically adds to navigation

## Contact Information
- **Email:** bombomatsitsa@gmail.com
- **Phone:** +254 702 816 978
- **GitHub:** https://github.com/Bombo9

For technical support or questions about the portfolio, use the contact form or reach out directly.