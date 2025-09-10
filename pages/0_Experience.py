import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Experience - Festus Bombo", page_icon="💼", layout="wide")

# Custom CSS for dark mode
st.markdown("""
<style>
    .experience-card {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 170, 0.2);
    }
    .company-name {
        font-size: 1.2rem;
        font-weight: bold;
        color: #00d4aa;
    }
    .position-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #fafafa;
        margin-bottom: 0.5rem;
    }
    .duration {
        color: #fafafa;
        opacity: 0.8;
        font-style: italic;
        margin-bottom: 1rem;
    }
    .education-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #00d4aa;
        border: 1px solid rgba(0, 212, 170, 0.3);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.1);
    }
    .timeline-dot {
        width: 12px;
        height: 12px;
        background-color: #00d4aa;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        box-shadow: 0 0 10px rgba(0, 212, 170, 0.5);
    }
</style>
""", unsafe_allow_html=True)

def calculate_duration(start_date, end_date="Present"):
    """Calculate duration between dates"""
    if end_date == "Present":
        end = datetime.now()
    else:
        end = datetime.strptime(end_date, "%b %Y")
    
    start = datetime.strptime(start_date, "%b %Y")
    months = (end.year - start.year) * 12 + (end.month - start.month)
    years = months // 12
    remaining_months = months % 12
    
    if years > 0 and remaining_months > 0:
        return f"{years} year{'s' if years > 1 else ''}, {remaining_months} month{'s' if remaining_months > 1 else ''}"
    elif years > 0:
        return f"{years} year{'s' if years > 1 else ''}"
    else:
        return f"{remaining_months} month{'s' if remaining_months > 1 else ''}"

def main():
    st.title("💼 Professional Experience")
    st.markdown("---")
    
    # Experience Timeline
    st.header("🚀 Career Journey")
    
    # Fiverr Experience
    st.markdown("""
    <div class="experience-card">
        <div class="position-title">🔹 Data Scientist</div>
        <div class="company-name">Fiverr</div>
        <div class="duration">April 2023 - Present ({})</div>
        <p><strong>Key Responsibilities & Achievements:</strong></p>
        <ul>
            <li>Built and maintained relationships with clients to understand their data needs</li>
            <li>Delivered custom data analysis solutions for diverse business requirements</li>
            <li>Developed predictive models and statistical analyses for client projects</li>
            <li>Created interactive dashboards and visualizations for business insights</li>
            <li>Provided data-driven recommendations to support client decision-making</li>
        </ul>
        <p><strong>Technologies Used:</strong> Python, R, SQL, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Tableau</p>
    </div>
    """.format(calculate_duration("Apr 2023")), unsafe_allow_html=True)
    
    # Upwork Experience
    st.markdown("""
    <div class="experience-card">
        <div class="position-title">🔹 Data Scientist</div>
        <div class="company-name">Upwork</div>
        <div class="duration">June 2022 - Present ({})</div>
        <p><strong>Key Responsibilities & Achievements:</strong></p>
        <ul>
            <li>Analyzed large datasets to identify trends and provide actionable insights</li>
            <li>Implemented machine learning algorithms for pattern recognition and prediction</li>
            <li>Performed comprehensive exploratory data analysis (EDA) for various industries</li>
            <li>Created automated data processing pipelines for efficient analysis</li>
            <li>Collaborated with international clients on complex data science projects</li>
        </ul>
        <p><strong>Technologies Used:</strong> Python, TensorFlow, Scikit-learn, Power BI, SQL, Excel, Statistical Analysis</p>
    </div>
    """.format(calculate_duration("Jun 2022")), unsafe_allow_html=True)
    
    # Skills developed through experience
    st.header("🎯 Skills Developed Through Experience")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💻 Technical Expertise")
        skills_technical = {
            "Python Programming": 95,
            "Data Analysis": 90,
            "Machine Learning": 85,
            "Statistical Modeling": 88,
            "SQL & Databases": 82,
            "Data Visualization": 90,
            "R Programming": 75,
            "Excel & Power BI": 85
        }
        
        for skill, level in skills_technical.items():
            st.write(f"**{skill}**")
            st.progress(level / 100)
            st.write(f"{level}%")
            st.write("")
    
    with col2:
        st.subheader("🤝 Professional Skills")
        skills_professional = {
            "Client Communication": 92,
            "Project Management": 88,
            "Problem Solving": 95,
            "Analytical Thinking": 93,
            "Team Collaboration": 85,
            "Adaptability": 90,
            "Time Management": 87,
            "Innovation": 88
        }
        
        for skill, level in skills_professional.items():
            st.write(f"**{skill}**")
            st.progress(level / 100)
            st.write(f"{level}%")
            st.write("")
    
    # Education Section
    st.header("🎓 Education Background")
    
    st.markdown("""
    <div class="education-card">
        <div class="position-title">🎓 Bachelor of Science in Computer Science</div>
        <div class="company-name">Pwani University</div>
        <div class="duration">August 2023 - September 2027 (Expected)</div>
        <p><strong>Current Status:</strong> Undergraduate Student</p>
        <p><strong>Relevant Coursework:</strong></p>
        <ul>
            <li>Data Structures and Algorithms</li>
            <li>Database Management Systems</li>
            <li>Statistical Methods and Analysis</li>
            <li>Machine Learning Fundamentals</li>
            <li>Software Engineering Principles</li>
            <li>Mathematical Modeling</li>
        </ul>
        <p><strong>Academic Focus:</strong> Combining theoretical computer science knowledge with practical data science applications</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Career Highlights
    st.header("🏆 Career Highlights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Years of Experience",
            value="3+",
            delta="Since 2021"
        )
    
    with col2:
        st.metric(
            label="Platforms",
            value="2 Major",
            delta="Fiverr & Upwork"
        )
    
    with col3:
        st.metric(
            label="Specialization",
            value="Data Science",
            delta="Full Stack"
        )
    
    # Professional Philosophy
    st.header("💡 Professional Philosophy")
    st.info("""
    **Commitment to Excellence:** I am committed to continuous learning, innovation, and delivering 
    high-impact, scalable analytical solutions in cross-functional team environments. My approach 
    combines strong technical skills with effective communication to transform complex data challenges 
    into clear, actionable business insights.
    """)

if __name__ == "__main__":
    main()
