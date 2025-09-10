import streamlit as st
import base64
from pathlib import Path
import os

st.set_page_config(
    page_title="Certificates - Festus Matsitsa",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Professional Certificates & Achievements")

# Introduction
st.markdown("""
This section showcases my professional certifications and achievements in data science, 
artificial intelligence, and related fields. Each certificate represents dedicated learning 
and practical application of skills.
""")

# Certificate data
certificates = [
    {
        "title": "Virtual Assistant Certificate",
        "organization": "ALX",
        "date": "October 2024",
        "duration": "8 weeks",
        "skills": ["Virtual Assistance Skills", "Digital Age Competencies", "Client Management"],
        "description": """Successfully completed an 8-week program in Virtual Assistance Skills 
        in the Digital Age. This comprehensive program covered essential skills for modern 
        virtual assistance including client relationship management and digital tools proficiency.""",
        "verification": "https://intranet.alxswe.com/certificates/m7zYRPMZXJ",
        "image_path": "attached_assets/72-virtual-assistant-certificate-festus-matsitsa.png"
    },
    {
        "title": "AI Career Essentials Certificate", 
        "organization": "ALX",
        "date": "August 2024",
        "duration": "8 weeks",
        "skills": ["AI Augmented Development", "Professional Skills", "Career Development"],
        "description": """Completed 8-week program in AI Augmented Professional Development Skills 
        in the Digital Age. Focused on integrating AI tools into professional workflows and 
        developing essential career skills for the AI era.""",
        "verification": "https://intranet.alxswe.com/certificates/NzRxHFZyYp",
        "image_path": "attached_assets/73-alx-aice-ai-career-essentials-certificate-festus-matsitsa (1).png"
    },
    {
        "title": "Data Science Job Simulation",
        "organization": "BCG via Forage",
        "date": "July 2024",
        "duration": "Project-based",
        "skills": ["Business Understanding", "Exploratory Data Analysis", "Feature Engineering", "Modeling"],
        "description": """Completed practical tasks in a real-world data science simulation including:
        Business Understanding & Hypothesis Framing, Exploratory Data Analysis, 
        Feature Engineering & Modelling, and Findings & Recommendations presentation.""",
        "verification": "Enrolment Code: qStTfL4xGp7nt3q6e | User Code: HLvAE4RYS53gvZqrq",
        "image_path": None
    },
    {
        "title": "GenAI Job Simulation",
        "organization": "BCG via Forage", 
        "date": "July 2024",
        "duration": "Project-based",
        "skills": ["Data Extraction", "AI Development", "Financial Chatbot", "GenAI Applications"],
        "description": """Completed practical tasks in Generative AI including:
        Data extraction and initial analysis, and developing an AI-powered financial chatbot 
        using cutting-edge GenAI technologies.""",
        "verification": "Enrolment Code: TxxdACbcnJp4NfR4K | User Code: HLvAE4RYS53gvZqrq",
        "image_path": None
    },
    {
        "title": "Data Science & Analytics",
        "organization": "HP LIFE",
        "date": "February 2025",
        "duration": "Self-paced",
        "skills": ["Data Science Practices", "Analytics Methodologies", "Business Applications"],
        "description": """Learned about leading data science and analytics practices, methodologies, 
        and tools. Examined the benefits and challenges of a data-driven approach for businesses, 
        and gained knowledge about essential skills needed to pursue a career in the field.""",
        "verification": "Certificate Serial: 2d674675-adb5-48d3-85b6-950fc3331632",
        "image_path": None
    },
    {
        "title": "Applied Data Science Lab",
        "organization": "WorldQuant University",
        "date": "August 2025",
        "duration": "Self-paced",
        "skills": ["Applied Data Science", "Research Methodology", "Statistical Analysis", "Data Mining"],
        "description": """Completed WorldQuant University's Applied Data Science Lab program, 
        focusing on practical application of data science techniques in real-world scenarios. 
        Gained hands-on experience in research methodology, statistical analysis, and advanced data mining techniques.""",
        "verification": "Certificate ID: WQU-ADSL-2025-08",
        "image_path": None
    }
]

# Certificate viewer modal
if 'show_cert' not in st.session_state:
    st.session_state.show_cert = None

# Full-screen certificate viewer
if st.session_state.show_cert:
    cert_to_show = st.session_state.show_cert
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown(f"### 🔍 {cert_to_show['title']} - Full View")
        if cert_to_show['image_path'] and os.path.exists(cert_to_show['image_path']):
            st.image(cert_to_show['image_path'], use_container_width=True)
        
        if st.button("❌ Close Full View", key="close_cert_view"):
            st.session_state.show_cert = None
            st.rerun()
    st.markdown("---")

# Display certificates
for i, cert in enumerate(certificates):
    with st.expander(f"🏆 {cert['title']} - {cert['organization']}", expanded=(i==0)):
        cert_col1, cert_col2 = st.columns([2, 1])
        
        with cert_col1:
            st.markdown(f"### 📋 Certificate Details")
            st.markdown(f"**Organization:** {cert['organization']}")
            st.markdown(f"**Date Completed:** {cert['date']}")
            st.markdown(f"**Duration:** {cert['duration']}")
            
            st.markdown(f"**Description:**")
            st.markdown(cert['description'])
            
            st.markdown(f"**Skills Acquired:**")
            for skill in cert['skills']:
                st.markdown(f"- {skill}")
            
            if 'verification' in cert:
                st.markdown(f"**Verification:** {cert['verification']}")
        
        with cert_col2:
            # Display certificate image if available
            if cert['image_path'] and os.path.exists(cert['image_path']):
                # Create clickable image with modal-like behavior
                st.image(cert['image_path'], caption=f"Click to view {cert['title']}", use_container_width=True)
                
                # Add a button to view in full size
                if st.button(f"🔍 View Full Certificate", key=f"view_{cert['title'].replace(' ', '_')}"):
                    st.session_state.show_cert = cert
                    st.rerun()
                    
            else:
                # Check if it's a PDF certificate
                pdf_path = None
                
                # Map certificates to their PDF files
                certificate_pdfs = {
                    "Data Science Job Simulation": "attached_assets/BCG DATA SCIENCE completion_certificate.pdf",
                    "GenAI Job Simulation": "attached_assets/BCG GEN AI_completion_certificate.pdf", 
                    "Data Science & Analytics": "attached_assets/hp certificate.pdf",
                    "Applied Data Science Lab": "attached_assets/AppliedDataScienceLab20250803-30-xbd0sf.pdf"
                }
                
                # Check for exact match first
                for cert_title, pdf_file in certificate_pdfs.items():
                    if cert_title.lower() in cert['title'].lower() or cert['title'].lower() in cert_title.lower():
                        if os.path.exists(pdf_file):
                            pdf_path = pdf_file
                            break
                
                # If no exact match, try organization matching
                if not pdf_path:
                    for pdf_file in certificate_pdfs.values():
                        if cert['organization'].upper() in pdf_file.upper():
                            if os.path.exists(pdf_file):
                                pdf_path = pdf_file
                                break
                
                if pdf_path:
                    st.markdown(f"""
                    <div style="
                        border: 2px solid #4ECDC4; 
                        padding: 2rem; 
                        text-align: center; 
                        border-radius: 10px;
                        background: linear-gradient(135deg, #4ECDC4, #45B7D1);
                        color: white;
                    ">
                        <h4>📄</h4>
                        <p><strong>PDF Certificate Available</strong></p>
                        <small>{cert['organization']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add download button for PDF
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        st.download_button(
                            label="📥 Download Certificate PDF",
                            data=pdf_bytes,
                            file_name=f"{cert['title'].replace(' ', '_')}_Certificate.pdf",
                            mime="application/pdf",
                            key=f"download_{cert['title'].replace(' ', '_')}"
                        )
                else:
                    # Placeholder for certificate image
                    st.markdown("""
                    <div style="
                        border: 2px dashed #4ECDC4; 
                        padding: 2rem; 
                        text-align: center; 
                        border-radius: 10px;
                        background: #f8f9fa;
                    ">
                        <h4>🎓</h4>
                        <p>Certificate Available</p>
                        <small>(Digital Verification)</small>
                    </div>
                    """, unsafe_allow_html=True)

# Summary statistics
st.markdown("---")
st.markdown("## 📊 Certification Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Certificates", len(certificates))

with col2:
    organizations = list(set([cert['organization'] for cert in certificates]))
    st.metric("Organizations", len(organizations))

with col3:
    total_skills = []
    for cert in certificates:
        total_skills.extend(cert['skills'])
    unique_skills = list(set(total_skills))
    st.metric("Unique Skills", len(unique_skills))

with col4:
    recent_cert = max(certificates, key=lambda x: x['date'])
    st.metric("Latest Certificate", recent_cert['date'])

# Skills gained from certifications
st.markdown("---")
st.markdown("## 🛠️ Skills Acquired Through Certifications")

all_skills = []
for cert in certificates:
    all_skills.extend(cert['skills'])

# Count skill frequency
skill_counts = {}
for skill in all_skills:
    skill_counts[skill] = skill_counts.get(skill, 0) + 1

# Display skills in columns
skill_col1, skill_col2, skill_col3 = st.columns(3)

skills_list = list(skill_counts.keys())
third = len(skills_list) // 3

with skill_col1:
    st.markdown("**Core Skills:**")
    for skill in skills_list[:third]:
        st.markdown(f"• {skill}")

with skill_col2:
    st.markdown("**Technical Skills:**")
    for skill in skills_list[third:third*2]:
        st.markdown(f"• {skill}")

with skill_col3:
    st.markdown("**Professional Skills:**")
    for skill in skills_list[third*2:]:
        st.markdown(f"• {skill}")

# Learning timeline
st.markdown("---")
st.markdown("## 📅 Learning Timeline")

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Convert dates to datetime for plotting
cert_timeline = []
for cert in certificates:
    try:
        # Simple date parsing - you may need to adjust based on actual date formats
        date_parts = cert['date'].split()
        if len(date_parts) >= 2:
            month_year = f"{date_parts[0]} {date_parts[1]}"
            cert_timeline.append({
                'Certificate': cert['title'],
                'Organization': cert['organization'],
                'Date': cert['date'],
                'Skills_Count': len(cert['skills'])
            })
    except:
        cert_timeline.append({
            'Certificate': cert['title'],
            'Organization': cert['organization'], 
            'Date': cert['date'],
            'Skills_Count': len(cert['skills'])
        })

timeline_df = pd.DataFrame(cert_timeline)

# Create timeline chart
fig = go.Figure()

for i, row in timeline_df.iterrows():
    fig.add_trace(go.Scatter(
        x=[i],
        y=[row['Skills_Count']],
        mode='markers+text',
        marker=dict(size=20, color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]),
        text=row['Certificate'],
        textposition="top center",
        name=row['Organization'],
        hovertemplate=f"<b>{row['Certificate']}</b><br>"
                     f"Organization: {row['Organization']}<br>"
                     f"Date: {row['Date']}<br>"
                     f"Skills: {row['Skills_Count']}<extra></extra>"
    ))

fig.update_layout(
    title="Certification Timeline & Skills Acquired",
    xaxis_title="Certification Order",
    yaxis_title="Number of Skills Acquired",
    height=400,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# Call to action
st.markdown("---")
st.markdown("""
### 🎯 Continuous Learning

I am committed to continuous professional development and regularly pursue new certifications 
and learning opportunities in emerging technologies and methodologies. Stay tuned for more 
certificates as I continue to expand my expertise in data science, AI, and related fields.

**Next Learning Goals:**
- Advanced Machine Learning Certification
- Cloud Platform Certifications (AWS/Azure)
- Deep Learning Specialization  
- Data Engineering Certification
""")
