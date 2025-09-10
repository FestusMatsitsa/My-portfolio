import streamlit as st
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="Contact & Messages",
    page_icon="📞",
    layout="wide"
)

def load_messages():
    """Load messages from JSON file"""
    if os.path.exists("data/messages.json"):
        try:
            with open("data/messages.json", "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_message(message_data):
    """Save a new message to JSON file"""
    messages = load_messages()
    messages.append(message_data)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    try:
        with open("data/messages.json", "w") as f:
            json.dump(messages, f, indent=2)
        return True
    except:
        return False

def main():
    st.title("📞 Contact & Get In Touch")
    st.markdown("Let's connect! I'm always open to discussing exciting opportunities, collaborations, and data science projects.")
    
    # Contact information section
    st.markdown("---")
    st.subheader("📧 Contact Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📱 Direct Contact
        
        **📧 Email:** [fmatsitsa@gmail.com](mailto:fmatsitsa@gmail.com)  
        **📱 Phone:** +254 702 816 978  
        **🐙 GitHub:** [github.com/FestusMatsitsa](https://github.com/FestusMatsitsa)  
        **🔗 LinkedIn:** Available upon request  
        
        ### 🌍 Location
        **📍 Based in:** Kenya  
        **🕐 Timezone:** EAT (UTC+3)  
        **💼 Available for:** Remote & On-site opportunities
        """)
    
    with col2:
        st.markdown("""
        ### 💼 I'm Open To:
        
        ✅ **Full-time Data Science Positions**  
        ✅ **Internships & Attachment Programs**  
        ✅ **Freelance Data Analytics Projects**  
        ✅ **Consulting Opportunities**  
        ✅ **Research Collaborations**  
        ✅ **Speaking Engagements**  
        
        ### 🎯 Areas of Interest:
        
        - Machine Learning & AI
        - Business Intelligence
        - Data Visualization
        - Statistical Analysis
        - Predictive Modeling
        """)
    
    # Message form section
    st.markdown("---")
    st.subheader("💬 Send Me a Message")
    st.markdown("Have a project in mind? Want to discuss opportunities? Feel free to reach out!")
    
    with st.form("contact_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Your Name *", placeholder="Enter your full name")
            email = st.text_input("Your Email *", placeholder="your.email@example.com")
        
        with col2:
            company = st.text_input("Company/Organization", placeholder="Your company (optional)")
            phone = st.text_input("Phone Number", placeholder="+254 XXX XXX XXX")
        
        subject = st.selectbox(
            "Subject *",
            [
                "Job Opportunity",
                "Internship Program",
                "Freelance Project", 
                "Collaboration Proposal",
                "Consultation Request",
                "General Inquiry",
                "Other"
            ]
        )
        
        if subject == "Other":
            custom_subject = st.text_input("Please specify:")
            subject = custom_subject if custom_subject else "Other"
        
        message = st.text_area(
            "Message *", 
            placeholder="Tell me about your project, opportunity, or how we can work together...",
            height=150
        )
        
        # Additional preferences
        st.markdown("**Preferred Contact Method:**")
        contact_pref = st.radio(
            "Preferred Contact Method",
            ["Email", "Phone", "Either"],
            horizontal=True
        )
        
        timeline = st.selectbox(
            "Timeline/Urgency",
            ["ASAP", "Within a week", "Within a month", "Flexible", "Just exploring"]
        )
        
        submitted = st.form_submit_button("📨 Send Message", use_container_width=True)
        
        if submitted:
            # Validation
            if not name or not email or not message:
                st.error("Please fill in all required fields (marked with *).")
            elif "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                # Create message object
                message_data = {
                    "timestamp": datetime.now().isoformat(),
                    "name": name,
                    "email": email,
                    "company": company,
                    "phone": phone,
                    "subject": subject,
                    "message": message,
                    "contact_preference": contact_pref,
                    "timeline": timeline
                }
                
                # Save message
                if save_message(message_data):
                    st.success("✅ Thank you! Your message has been sent successfully. I'll get back to you soon!")
                    st.balloons()
                else:
                    st.warning("⚠️ There was an issue saving your message, but don't worry! Please reach out directly via email: bombomatsitsa@gmail.com")
    
    # FAQ section
    st.markdown("---")
    st.subheader("❓ Frequently Asked Questions")
    
    with st.expander("What types of projects do you work on?"):
        st.markdown("""
        I work on a wide variety of data science projects including:
        - **Business Analytics:** Sales forecasting, customer segmentation, market analysis
        - **Machine Learning:** Predictive modeling, classification, clustering
        - **Data Visualization:** Interactive dashboards, reports, insights presentation
        - **Statistical Analysis:** A/B testing, hypothesis testing, experimental design
        - **Process Automation:** Data pipelines, automated reporting, ETL processes
        """)
    
    with st.expander("What is your typical project timeline?"):
        st.markdown("""
        Project timelines vary based on complexity and scope:
        - **Quick Analysis:** 1-3 days for basic EDA and insights
        - **Dashboard Development:** 1-2 weeks for comprehensive dashboards
        - **ML Model Development:** 2-4 weeks including data prep, training, and validation
        - **Full Analytics Solutions:** 1-3 months for end-to-end implementations
        
        I always provide realistic timelines during our initial discussion.
        """)
    
    with st.expander("Do you offer remote work?"):
        st.markdown("""
        Yes! I'm fully equipped for remote work and have experience collaborating with international teams.
        I can work across different time zones and am comfortable with various collaboration tools.
        
        I'm also open to hybrid arrangements and on-site work for local opportunities.
        """)
    
    with st.expander("What are your rates for freelance work?"):
        st.markdown("""
        My rates depend on several factors:
        - Project complexity and scope
        - Timeline and urgency
        - Long-term vs. short-term engagement
        - Technology stack requirements
        
        I offer competitive rates and am happy to discuss pricing during our initial consultation.
        I also provide fixed-price quotes for well-defined projects.
        """)
    
    # Call to action
    st.markdown("---")
    st.markdown("### 🚀 Ready to Start Your Data Science Journey?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Data Analysis**
        Transform your raw data into actionable insights with comprehensive analysis and visualization.
        """)
    
    with col2:
        st.markdown("""
        **🤖 Machine Learning**
        Build predictive models and intelligent systems to automate decisions and forecast trends.
        """)
    
    with col3:
        st.markdown("""
        **📊 Business Intelligence**
        Create interactive dashboards and reporting systems for real-time business monitoring.
        """)
    
    st.markdown("---")
    st.markdown("*Looking forward to hearing from you and discussing how we can work together to achieve your data science goals!* 🎯")

if __name__ == "__main__":
    main()
