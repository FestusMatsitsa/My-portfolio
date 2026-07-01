import streamlit as st
import base64
import os
from io import BytesIO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Festus Matsitsa Bombo - Data Scientist",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="auto"
)
# Inject Google site verification meta tag
st.markdown(
    """
    <meta name="google-site-verification" content="gCTDmrgKgrPw_5Wh9EKQQWsDxmvSAxzzeIyps7MWC5A" />
    """,
    unsafe_allow_html=True
)


# Custom CSS for dark mode styling with mobile responsiveness
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
        }
        .sub-header {
            font-size: 1.1rem !important;
        }
        .section-header {
            font-size: 1.5rem !important;
        }
    }
    
    /* Typography */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #00d4aa;
        text-shadow: 0 0 10px rgba(0, 212, 170, 0.3);
        transition: all 0.3s ease;
    }
    
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #fafafa;
        margin-bottom: 2rem;
        opacity: 0.9;
        letter-spacing: 2px;
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #00d4aa;
        padding-bottom: 0.5rem;
        color: #00d4aa;
        text-shadow: 0 0 8px rgba(0, 212, 170, 0.2);
    }
    
    /* Profile Image Border */
    .profile-image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        perspective: 1000px;
    }
    
    .profile-image-border {
        position: relative;
        border-radius: 50%;
        overflow: hidden;
        box-shadow: 
            0 0 0 3px #0e1117,
            0 0 0 6px #00d4aa,
            0 0 30px rgba(0, 212, 170, 0.5),
            0 0 60px rgba(0, 212, 170, 0.3),
            0 8px 32px rgba(0, 0, 0, 0.4);
        animation: glow 3s ease-in-out infinite;
        transition: all 0.3s ease;
    }
    
    .profile-image-border:hover {
        box-shadow: 
            0 0 0 3px #0e1117,
            0 0 0 6px #00d4aa,
            0 0 50px rgba(0, 212, 170, 0.7),
            0 0 80px rgba(0, 212, 170, 0.4),
            0 12px 48px rgba(0, 0, 0, 0.5);
        transform: scale(1.05);
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 
                0 0 0 3px #0e1117,
                0 0 0 6px #00d4aa,
                0 0 30px rgba(0, 212, 170, 0.5),
                0 0 60px rgba(0, 212, 170, 0.3),
                0 8px 32px rgba(0, 0, 0, 0.4);
        }
        50% {
            box-shadow: 
                0 0 0 3px #0e1117,
                0 0 0 6px #00d4aa,
                0 0 50px rgba(0, 212, 170, 0.7),
                0 0 80px rgba(0, 212, 170, 0.4),
                0 8px 40px rgba(0, 0, 0, 0.5);
        }
    }
    
    .profile-image-border img {
        display: block;
        width: 100%;
        height: auto;
    }
    
    .profile-label {
        text-align: center;
        opacity: 0.8;
        font-size: 0.9rem;
        margin-top: 1rem;
        color: #00d4aa;
        font-weight: 500;
    }
    
    /* Navigation Styles */
    .nav-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 2px solid rgba(0, 212, 170, 0.3);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.1);
    }
    
    .nav-title {
        color: #00d4aa;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.8rem;
    }
    
    .nav-description {
        color: #fafafa;
        opacity: 0.85;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Resume Button */
    .resume-button-container {
        display: flex;
        justify-content: center;
        margin: 1.5rem 0;
    }
    
    .resume-download-link {
        background: linear-gradient(135deg, #00d4aa 0%, #00b894 100%);
        color: #0e1117 !important;
        text-decoration: none !important;
        font-weight: bold;
        padding: 1rem 2rem;
        border: 3px solid #00d4aa;
        border-radius: 30px;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    
    .resume-download-link:hover {
        background: linear-gradient(135deg, #00b894 0%, #009977 100%);
        border-color: #00b894;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.5);
    }
    
    /* Card Styles */
    .highlight-box {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 170, 0.2);
        transition: all 0.3s ease;
    }
    
    .highlight-box:hover {
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.2);
        border: 1px solid rgba(0, 212, 170, 0.4);
    }
    
    .contact-info {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid rgba(0, 212, 170, 0.3);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.1);
        transition: all 0.3s ease;
    }
    
    .contact-info:hover {
        border: 2px solid rgba(0, 212, 170, 0.6);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.2);
    }
    
    .project-card {
        background: linear-gradient(135deg, #2a2a3e 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 170, 0.2);
        transition: all 0.3s ease;
    }
    
    .project-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 30px rgba(0, 212, 170, 0.25);
        border: 1px solid rgba(0, 212, 170, 0.5);
    }
    
    .contact-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 3px solid rgba(0, 212, 170, 0.4);
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.1);
        transition: all 0.3s ease;
    }
    
    .contact-section:hover {
        border: 3px solid rgba(0, 212, 170, 0.6);
        box-shadow: 0 12px 40px rgba(0, 212, 170, 0.15);
    }
    
    /* Tech Badge */
    .tech-badge {
        background: linear-gradient(45deg, #00d4aa, #00b894);
        color: #0e1117 !important;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        margin: 0.3rem;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: bold;
        box-shadow: 0 3px 10px rgba(0, 212, 170, 0.3);
        transition: all 0.3s ease;
        border: 2px solid rgba(0, 212, 170, 0.5);
    }
    
    .tech-badge:hover {
        transform: scale(1.1);
        box-shadow: 0 5px 15px rgba(0, 212, 170, 0.5);
    }
    
    /* Contact Links */
    .email-link {
        color: #00d4aa !important;
        text-decoration: none !important;
        font-weight: bold;
        padding: 0.7rem 1.3rem;
        border: 2px solid #00d4aa;
        border-radius: 30px;
        display: inline-block;
        transition: all 0.3s ease;
        background: rgba(0, 212, 170, 0.1);
        margin: 0.5rem;
        font-size: 0.95rem;
    }
    
    .email-link:hover {
        background: #00d4aa;
        color: #0e1117 !important;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
    }
    
    /* Responsive Layout */
    @media (max-width: 1024px) {
        .main-content {
            flex-direction: column;
        }
    }
    
    @media (max-width: 768px) {
        .profile-image-border {
            width: 200px;
            height: 200px;
        }
        
        .main-header {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        
        .section-header {
            font-size: 1.3rem;
            margin-top: 1.5rem;
        }
        
        .contact-section {
            padding: 1.5rem;
            margin: 1.5rem 0;
        }
        
        .resume-download-link {
            padding: 0.8rem 1.5rem;
            font-size: 0.9rem;
        }
        
        .email-link {
            padding: 0.6rem 1rem;
            font-size: 0.85rem;
            margin: 0.3rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.5rem;
        }
        
        .sub-header {
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
        }
        
        .section-header {
            font-size: 1.1rem;
            margin-top: 1rem;
        }
        
        .profile-image-border {
            width: 150px;
            height: 150px;
        }
        
        .profile-label {
            font-size: 0.8rem;
        }
        
        .contact-section {
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .tech-badge {
            padding: 0.3rem 0.7rem;
            font-size: 0.75rem;
            margin: 0.2rem;
        }
        
        .email-link {
            padding: 0.5rem 0.8rem;
            font-size: 0.8rem;
            margin: 0.2rem;
        }
        
        .resume-download-link {
            padding: 0.7rem 1.2rem;
            font-size: 0.85rem;
            border: 2px solid #00d4aa;
        }
    }
    
    /* Smooth Scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Focus States for Accessibility */
    a:focus, button:focus {
        outline: 2px solid #00d4aa;
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)

def create_download_link():
    """Create a download link for the resume PDF"""
    try:
        with open("assets/resume.pdf", "rb") as file:
            pdf_data = file.read()
            b64 = base64.b64encode(pdf_data).decode()
            href = f'<div class="resume-button-container"><a href="data:application/pdf;base64,{b64}" download="Festus_Bombo_Resume.pdf" class="resume-download-link">📄 Download Resume (PDF)</a></div>'
            return href
    except FileNotFoundError:
        return '<div class="resume-button-container"><p style="color: #ff6b6b;">📄 Resume file not available for download</p></div>'

def main():
    # Header
    st.markdown('<h1 class="main-header">FESTUS MATSITSA BOMBO</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">🚀 DATA SCIENTIST</p>', unsafe_allow_html=True)
    
    # Profile image section with enhanced border
    col1, col2, col3 = st.columns([1, 2, 1], gap="large")
    with col2:
        if os.path.exists("assets/profile.png"):
            image = Image.open("assets/profile.png")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            st.markdown(
                f"""
                <div class="profile-image-container">
                    <div class="profile-image-border">
                        <img src="data:image/png;base64,{img_b64}" alt="Festus Matsitsa Bombo" />
                    </div>
                </div>
                <p class="profile-label">✨ Festus Matsitsa Bombo - Data Scientist</p>
                """,
                unsafe_allow_html=True
            )
    
    # Improved Sidebar Navigation
    with st.sidebar:
        st.markdown('<div class="nav-container">', unsafe_allow_html=True)
        st.markdown('<div class="nav-title">🧭 Navigation Hub</div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-description">Explore the sections below to discover my experience, skills, projects, and more!</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        
        # Quick Links
        st.sidebar.markdown("### 📑 Quick Links")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💼 Experience", use_container_width=True):
                st.switch_page("pages/0_Experience.py")
            if st.button("🎯 Skills", use_container_width=True):
                st.switch_page("pages/1_skills.py")
            if st.button("🎓 Certificates", use_container_width=True):
                st.switch_page("pages/2_Certificates.py")
        
        with col2:
            if st.button("🔬 Toolkit", use_container_width=True):
                st.switch_page("pages/3_Data_Science_Toolkit.py")
            if st.button("🚀 Projects", use_container_width=True):
                st.switch_page("pages/4_Projects.py")
            if st.button("📧 Contact", use_container_width=True):
                st.switch_page("pages/5_Contact_Me.py")
        
        st.sidebar.markdown("---")
        
        # Resume Download in Sidebar
        st.sidebar.markdown("### 📄 Resume")
        st.sidebar.markdown(create_download_link(), unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        
        # Quick Info
        st.sidebar.markdown("### 🌟 Quick Info")
        st.sidebar.info("""
        📍 **Location**: Kenya  
        🎓 **Education**: BSc Computer Science  
        💼 **Status**: Freelance Data Scientist  
        ⭐ **Experience**: 3+ Years  
        🌐 **Available for**: Remote & On-site  
        """)
    
    st.markdown("---")

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
    
    # Main content with responsive layout
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        st.markdown('<h2 class="section-header">👨‍💻 About Me</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
        <p><strong>Passionate Data Scientist</strong> with a strong background in data analysis, machine learning, 
        data visualization, and statistical modeling. I specialize in transforming complex datasets 
        into actionable insights that support data-driven decision-making and drive business success.</p>
        
        <p>Proficient in <strong>Python, R, SQL, Excel, Power BI, and Tableau</strong>, with hands-on experience 
        using libraries such as Pandas, NumPy, Scikit-learn, and TensorFlow. Experienced in 
        developing predictive models, performing exploratory data analysis (EDA), A/B testing, 
        data cleaning, and applying advanced statistical methods to solve real-world problems.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">🎯 Core Competencies</h2>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2, gap="small")
        
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
