import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Skills - Festus Bombo", page_icon="🛠️", layout="wide")

# Custom CSS for dark mode
st.markdown("""
<style>
    .skill-category {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 170, 0.2);
    }
    .tool-badge {
        background: linear-gradient(45deg, #00d4aa, #00b894);
        color: #0e1117;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0, 212, 170, 0.3);
        border: 1px solid #00d4aa;
    }
    .proficiency-high { color: #00d4aa; font-weight: bold; }
    .proficiency-medium { color: #ffc107; font-weight: bold; }
    .proficiency-basic { color: #6c757d; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def create_skills_radar_chart():
    """Create a radar chart for technical skills"""
    skills = ['Python', 'R', 'SQL', 'Machine Learning', 'Statistics', 'Data Viz', 'Excel', 'Cloud Tech']
    values = [95, 75, 85, 88, 90, 92, 85, 70]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=skills,
        fill='toself',
        name='Technical Skills',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Technical Skills Proficiency",
        height=400
    )
    
    return fig

def create_tools_treemap():
    """Create a treemap for tools and technologies"""
    tools_data = {
        'Category': ['Programming', 'Programming', 'Programming', 'ML/AI', 'ML/AI', 'ML/AI', 
                    'Visualization', 'Visualization', 'Visualization', 'Database', 'Database', 'Cloud'],
        'Tool': ['Python', 'R', 'SQL', 'Scikit-learn', 'TensorFlow', 'Pandas', 
                'Tableau', 'Power BI', 'Matplotlib', 'MySQL', 'PostgreSQL', 'AWS'],
        'Proficiency': [95, 75, 85, 88, 80, 92, 88, 85, 90, 80, 75, 70],
        'Experience': [3.5, 2.5, 3.0, 2.8, 2.0, 3.2, 2.5, 2.8, 3.0, 2.5, 2.0, 1.5]
    }
    
    df = pd.DataFrame(tools_data)
    
    fig = px.treemap(df, path=['Category', 'Tool'], values='Proficiency',
                     color='Experience', hover_data=['Proficiency'],
                     color_continuous_scale='Blues',
                     title="Tools & Technologies Proficiency")
    
    return fig

def main():
    st.title("🛠️ Skills & Expertise")
    st.markdown("---")
    
    # Overview
    st.header("📊 Skills Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(create_skills_radar_chart(), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_tools_treemap(), use_container_width=True)
    
    # Detailed Skills Breakdown
    st.header("🔧 Detailed Skills Breakdown")
    
    # Programming Languages
    st.subheader("💻 Programming Languages")
    st.markdown("""
    <div class="skill-category">
        <h4>Core Languages</h4>
        <div>
            <span class="tool-badge">🐍 Python (Advanced)</span>
            <span class="tool-badge">📊 R (Intermediate)</span>
            <span class="tool-badge">🗄️ SQL (Advanced)</span>
        </div>
        <br>
        <p><strong>Python Libraries:</strong> Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib, Seaborn, Plotly, Scipy, Statsmodels</p>
        <p><strong>R Packages:</strong> dplyr, ggplot2, caret, randomForest, tidyverse</p>
        <p><strong>SQL Variants:</strong> MySQL, PostgreSQL, SQLite</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Science & Analytics
    st.subheader("🔬 Data Science & Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Machine Learning:**
        - Supervised Learning (Classification, Regression)
        - Unsupervised Learning (Clustering, PCA)
        - Model Selection & Validation
        - Feature Engineering & Selection
        - Hyperparameter Tuning
        - Ensemble Methods
        """)
        
        st.markdown("""
        **Statistical Analysis:**
        - Descriptive & Inferential Statistics
        - Hypothesis Testing
        - A/B Testing
        - Time Series Analysis
        - Bayesian Statistics
        - Experimental Design
        """)
    
    with col2:
        st.markdown("""
        **Data Processing:**
        - Data Cleaning & Preprocessing
        - Data Transformation & Normalization
        - Handling Missing Data
        - Outlier Detection & Treatment
        - Data Pipeline Development
        - ETL Processes
        """)
        
        st.markdown("""
        **Deep Learning:**
        - Neural Networks
        - TensorFlow & Keras
        - Computer Vision Basics
        - Natural Language Processing
        - Model Optimization
        - Transfer Learning
        """)
    
    # Visualization & Reporting Tools
    st.subheader("📈 Visualization & Reporting")
    st.markdown("""
    <div class="skill-category">
        <h4>Visualization Tools</h4>
        <div>
            <span class="tool-badge">📊 Tableau (Advanced)</span>
            <span class="tool-badge">📈 Power BI (Advanced)</span>
            <span class="tool-badge">🎨 Matplotlib (Expert)</span>
            <span class="tool-badge">🌊 Seaborn (Expert)</span>
            <span class="tool-badge">📱 Plotly (Advanced)</span>
            <span class="tool-badge">📋 Excel (Advanced)</span>
        </div>
        <br>
        <p><strong>Specializations:</strong></p>
        <ul>
            <li>Interactive Dashboard Development</li>
            <li>Business Intelligence Reporting</li>
            <li>Statistical Visualization</li>
            <li>Geospatial Data Visualization</li>
            <li>Web-based Interactive Charts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Skills
    st.subheader("🤝 Professional & Soft Skills")
    
    professional_skills = {
        "Communication": {"level": 92, "description": "Clear presentation of complex technical concepts to non-technical stakeholders"},
        "Problem Solving": {"level": 95, "description": "Systematic approach to identifying and solving data-related challenges"},
        "Project Management": {"level": 88, "description": "Managing multiple client projects with varying timelines and requirements"},
        "Analytical Thinking": {"level": 93, "description": "Breaking down complex problems into manageable, logical components"},
        "Collaboration": {"level": 85, "description": "Working effectively in cross-functional teams and remote environments"},
        "Adaptability": {"level": 90, "description": "Quickly learning new tools and techniques as technology evolves"},
        "Attention to Detail": {"level": 94, "description": "Ensuring accuracy and quality in data analysis and model development"},
        "Time Management": {"level": 87, "description": "Efficiently managing multiple projects and meeting deadlines"}
    }
    
    for skill, details in professional_skills.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(skill, f"{details['level']}%")
        with col2:
            st.progress(details['level'] / 100)
            st.caption(details['description'])
    
    # Certifications & Learning
    st.header("🏆 Certifications & Continuous Learning")
    
    st.info("""
    **Commitment to Continuous Learning:**
    As a data scientist, I stay current with the latest developments in the field through:
    - Online courses and specializations
    - Industry conferences and webinars
    - Open source contributions
    - Personal projects and experiments
    - Professional networking and knowledge sharing
    """)
    
    # Domain Expertise
    st.header("🎯 Domain Expertise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Industry Experience:**
        - E-commerce Analytics
        - Financial Data Analysis
        - Healthcare Data Processing
        - Marketing Analytics
        - Operations Research
        """)
    
    with col2:
        st.markdown("""
        **Specialized Applications:**
        - Predictive Modeling
        - Customer Segmentation
        - Fraud Detection
        - Recommendation Systems
        - Business Intelligence
        """)
    
    # Skills Assessment
    st.header("📋 Self-Assessment Matrix")
    
    assessment_data = {
        'Skill Area': ['Data Collection', 'Data Cleaning', 'EDA', 'Statistical Analysis', 
                      'Machine Learning', 'Data Visualization', 'Model Deployment', 'Communication'],
        'Proficiency': [85, 92, 95, 90, 88, 93, 75, 92],
        'Experience (Years)': [3.5, 3.5, 3.5, 3.0, 2.8, 3.2, 2.0, 3.5],
        'Project Count': [50, 50, 50, 45, 35, 48, 15, 50]
    }
    
    df_assessment = pd.DataFrame(assessment_data)
    
    fig = px.scatter(df_assessment, x='Experience (Years)', y='Proficiency', 
                     size='Project Count', hover_name='Skill Area',
                     title="Skills Experience vs Proficiency Matrix",
                     labels={'Proficiency': 'Proficiency Level (%)', 'Experience (Years)': 'Years of Experience'})
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
