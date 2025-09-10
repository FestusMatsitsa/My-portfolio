import streamlit as st
import base64
import os
from io import BytesIO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Festus Matsitsa Bombo - Data Scientist",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #00d4aa;
        text-shadow: 0 0 10px rgba(0, 212, 170, 0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #fafafa;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #00d4aa;
        padding-bottom: 0.5rem;
        color: #00d4aa;
    }
    .highlight-box {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 170, 0.2);
    }
    .contact-info {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(0, 212, 170, 0.3);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.1);
    }
    .project-card {
        background: linear-gradient(135deg, #2a2a3e 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 170, 0.2);
        transition: transform 0.3s ease;
    }
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.2);
    }
    .contact-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 2px solid rgba(0, 212, 170, 0.3);
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.1);
    }
    .tech-badge {
        background: linear-gradient(45deg, #00d4aa, #00b894);
        color: #0e1117;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0, 212, 170, 0.3);
    }
    .email-link {
        color: #00d4aa;
        text-decoration: none;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border: 2px solid #00d4aa;
        border-radius: 25px;
        display: inline-block;
        transition: all 0.3s ease;
        background: rgba(0, 212, 170, 0.1);
    }
    .email-link:hover {
        background: #00d4aa;
        color: #0e1117;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

def create_download_link():
    """Create a download link for the resume PDF"""
    try:
        with open("assets/resume.pdf", "rb") as file:
            pdf_data = file.read()
            b64 = base64.b64encode(pdf_data).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="Festus_Bombo_Resume.pdf">📄 Download Resume (PDF)</a>'
            return href
    except FileNotFoundError:
        return '<p>Resume file not available for download</p>'

def main():
    # Header
    st.markdown('<h1 class="main-header">FESTUS MATSITSA BOMBO</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">DATA SCIENTIST</p>', unsafe_allow_html=True)
    
    # Profile image section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if os.path.exists("assets/profile.png"):
            image = Image.open("assets/profile.png")
            st.image(image, width=300, caption="Festus Matsitsa Bombo - Data Scientist")
    
    # Navigation info
    st.sidebar.title("🧭 Navigation")
    st.sidebar.info("Use the pages in the sidebar to explore different sections of the portfolio.")
    
    # Resume download
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Resume")
    st.sidebar.markdown(create_download_link(), unsafe_allow_html=True)

    # Certificate Gallery Section
    st.markdown("### 🎓 Professional Certificates")
    
    # Create certificate cards
    certificates = [
        {
            "title": "WorldQuant University Applied Data Science Lab",
            "organization": "WorldQuant University",
            "date": "August 2025",
            "skills": "Applied Data Science, Research Methodology",
            "color": "#FF6B6B"
        },
        {
            "title": "HP LIFE Data Science & Analytics",
            "organization": "HP LIFE",
            "date": "February 2025", 
            "skills": "Data Science Practices, Analytics",
            "color": "#4ECDC4"
        },
        {
            "title": "ALX Virtual Assistant Certificate",
            "organization": "ALX",
            "date": "October 2024",
            "skills": "Virtual Assistance, Digital Skills",
            "color": "#45B7D1"
        },
        {
            "title": "ALX AI Career Essentials",
            "organization": "ALX", 
            "date": "August 2024",
            "skills": "AI Development, Professional Skills",
            "color": "#96CEB4"
        },
        {
            "title": "BCG Data Science Simulation",
            "organization": "BCG via Forage",
            "date": "July 2024",
            "skills": "Business Understanding, ML Modeling",
            "color": "#FECA57"
        },
        {
            "title": "BCG GenAI Simulation", 
            "organization": "BCG via Forage",
            "date": "July 2024",
            "skills": "AI Development, Financial Chatbots",
            "color": "#FF9FF3"
        }
    ]
    
    # Display certificates in a grid
    cert_col1, cert_col2, cert_col3 = st.columns(3)
    
    for i, cert in enumerate(certificates):
        col = [cert_col1, cert_col2, cert_col3][i % 3]
        
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {cert['color']}, {cert['color']}AA);
                color: white;
                padding: 1.5rem;
                border-radius: 15px;
                margin: 0.5rem 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div>
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">{cert['title']}</h4>
                    <p style="margin: 0.5rem 0; font-size: 0.8rem; opacity: 0.9;"><strong>{cert['organization']}</strong></p>
                </div>
                <div>
                    <p style="margin: 0.5rem 0; font-size: 0.75rem; opacity: 0.8;">{cert['skills']}</p>
                    <p style="margin: 0; font-size: 0.8rem; font-weight: bold;">{cert['date']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # View all certificates button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <p>🔍 <strong>Want to see detailed certificate information?</strong></p>
            <p>Visit the <strong>📋 Certificates</strong> page for verification codes, detailed descriptions, and skills acquired.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">👨‍💻 About Me</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
        <p>Passionate Data Scientist with a strong background in data analysis, machine learning, 
        data visualization, and statistical modeling. I specialize in transforming complex datasets 
        into actionable insights that support data-driven decision-making and drive business success.</p>
        
        <p>Proficient in Python, R, SQL, Excel, Power BI, and Tableau, with hands-on experience 
        using libraries such as Pandas, NumPy, Scikit-learn, and TensorFlow. Experienced in 
        developing predictive models, performing exploratory data analysis (EDA), A/B testing, 
        data cleaning, and applying advanced statistical methods to solve real-world problems.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">🎯 Core Competencies</h2>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
            **Technical Skills:**
            - 🐍 Python & R Programming
            - 📊 Data Visualization (Tableau, Power BI)
            - 🤖 Machine Learning & Deep Learning
            - 📈 Statistical Analysis & Modeling
            - 🗄️ SQL & Database Management
            """)
        
        with col_b:
            st.markdown("""
            **Soft Skills:**
            - 🧠 Analytical Thinking
            - 🔍 Attention to Detail
            - 🗣️ Communication & Collaboration
            - ⚡ Adaptability & Innovation
            - 📋 Project Management
            """)
    
    with col2:
        st.markdown('<h2 class="section-header">📞 Contact</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="contact-info">
        <p><strong>🏫 Education:</strong><br>
        BSc Computer Science<br>
        Pwani University<br>
        Aug 2023 - Sep 2027</p>
        
        <p><strong>💼 Current Status:</strong><br>
        Freelance Data Scientist<br>
        Fiverr & Upwork</p>
        
        <p><strong>🌟 Experience:</strong><br>
        3+ years in Data Science<br>
        Specialized in client solutions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent Projects Section
    st.markdown('<h2 class="section-header">🚀 Featured Projects</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="project-card">
        <h4>📊 Customer Segmentation Analysis</h4>
        <p>Developed a comprehensive customer segmentation model using K-means clustering and RFM analysis for an e-commerce client, resulting in 25% increase in targeted marketing effectiveness.</p>
        <div>
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">Tableau</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="project-card">
        <h4>🤖 Sales Forecasting Model</h4>
        <p>Built time series forecasting models using ARIMA and Random Forest to predict monthly sales with 92% accuracy, helping a retail client optimize inventory management.</p>
        <div>
        <span class="tech-badge">Python</span>
        <span class="tech-badge">TensorFlow</span>
        <span class="tech-badge">Time Series</span>
        <span class="tech-badge">Power BI</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="project-card">
        <h4>📈 Financial Risk Assessment</h4>
        <p>Developed a credit risk assessment model using logistic regression and gradient boosting, achieving 88% accuracy in predicting loan defaults for a fintech startup.</p>
        <div>
        <span class="tech-badge">Python</span>
        <span class="tech-badge">XGBoost</span>
        <span class="tech-badge">SQL</span>
        <span class="tech-badge">Excel</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="project-card">
        <h4>🔍 Market Analysis Dashboard</h4>
        <p>Created an interactive dashboard analyzing market trends and competitor performance using web scraping and data visualization, providing real-time insights for strategic decision-making.</p>
        <div>
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">Web Scraping</span>
        <span class="tech-badge">Dashboard</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Contact & Opportunities Section
    st.markdown('<h2 class="section-header">💼 Open for Opportunities</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="contact-section">
    <h3 style="color: #00d4aa; margin-bottom: 1.5rem; text-align: center;">🚀 Ready to Collaborate</h3>
    
    <div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.1rem; margin-bottom: 1rem;">I'm actively seeking opportunities in:</p>
    <div style="margin-bottom: 1.5rem;">
    <span class="tech-badge">Data Science Internships</span>
    <span class="tech-badge">Full-time Positions</span>
    <span class="tech-badge">Freelance Projects</span>
    <span class="tech-badge">Research Collaborations</span>
    </div>
    </div>
    
    <div style="text-align: center; margin-bottom: 2rem;">
    <h4 style="color: #00d4aa; margin-bottom: 1rem;">📞 Contact Information</h4>
    <div style="margin-bottom: 1rem;">
    <a href="mailto:fmatsitsa@gmail.com" class="email-link" style="margin: 0.5rem;">
    📧 fmatsitsa@gmail.com
    </a>
    </div>
    <div style="margin-bottom: 1rem;">
    <a href="tel:+254702816978" class="email-link" style="margin: 0.5rem;">
    📱 +254 702 816 978
    </a>
    </div>
    <div>
    <a href="https://github.com/FestusMatsitsa/FestusMatsitsa" target="_blank" class="email-link" style="margin: 0.5rem;">
    🔗 GitHub Portfolio
    </a>
    </div>
    </div>
    
    <div style="text-align: center;">
    <p style="opacity: 0.9; margin-bottom: 0.5rem;">Available for:</p>
    <p style="margin: 0;">✓ Remote work • ✓ On-site opportunities • ✓ Contract projects • ✓ Team collaborations</p>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    st.markdown("### 🔧 Explore My Data Science Toolkit")
    st.info("Navigate to the **Data Science Toolkit** page to see interactive tools for data analysis, visualization, and machine learning!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #fafafa; opacity: 0.8;'>
    <p>© 2024 Festus Matsitsa Bombo | Data Scientist Portfolio</p>
    <p>Committed to continuous learning, innovation, and delivering high-impact, scalable analytical solutions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
