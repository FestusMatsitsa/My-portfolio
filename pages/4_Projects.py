import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Projects Portfolio",
    page_icon="📊",
    layout="wide"
)

def create_sample_data():
    """Create sample data for project demonstrations"""
    # Sample sales data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
        'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'product': np.random.choice(['Product A', 'Product B', 'Product C'], len(dates))
    })
    sales_data['sales'] = np.maximum(sales_data['sales'], 0)
    
    # Customer segmentation data
    customer_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'spending_score': np.random.normal(50, 20, 1000),
        'gender': np.random.choice(['Male', 'Female'], 1000)
    })
    
    # Stock price data
    stock_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2024-01-01', freq='D'),
        'price': 100 + np.cumsum(np.random.normal(0, 2, pd.date_range('2020-01-01', '2024-01-01', freq='D').shape[0]))
    })
    
    return sales_data, customer_data, stock_data

def sales_analysis_project():
    """Sales Analysis and Forecasting Project"""
    st.header("📈 Sales Analysis & Forecasting")
    
    st.markdown("""
    **Project Overview:**
    This project demonstrates comprehensive sales data analysis including trend identification,
    seasonal patterns, regional performance, and basic forecasting capabilities.
    
    **Technologies Used:** Python, Pandas, Plotly, Statistical Analysis
    
    **Key Features:**
    - Time series analysis
    - Regional performance comparison
    - Product performance analysis
    - Trend identification
    """)
    
    sales_data, _, _ = create_sample_data()
    
    # Time series analysis
    st.subheader("📅 Sales Trends Over Time")
    
    # Aggregate daily sales
    daily_sales = sales_data.groupby('date')['sales'].sum().reset_index()
    
    fig = px.line(daily_sales, x='date', y='sales', 
                  title="Daily Sales Trend",
                  labels={'sales': 'Sales ($)', 'date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional analysis
    st.subheader("🗺️ Regional Performance")
    
    regional_sales = sales_data.groupby('region')['sales'].agg(['sum', 'mean', 'count']).reset_index()
    regional_sales.columns = ['Region', 'Total Sales', 'Average Sales', 'Number of Transactions']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(regional_sales, x='Region', y='Total Sales', 
                     title="Total Sales by Region")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(regional_sales, values='Total Sales', names='Region', 
                     title="Sales Distribution by Region")
        st.plotly_chart(fig, use_container_width=True)
    
    # Product analysis
    st.subheader("📦 Product Performance")
    
    product_sales = sales_data.groupby('product')['sales'].agg(['sum', 'mean']).reset_index()
    product_sales.columns = ['Product', 'Total Sales', 'Average Sales']
    
    fig = px.bar(product_sales, x='Product', y=['Total Sales', 'Average Sales'], 
                 title="Product Performance Comparison", barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trends
    st.subheader("📊 Monthly Sales Patterns")
    
    sales_data['month'] = sales_data['date'].dt.month
    monthly_sales = sales_data.groupby('month')['sales'].sum().reset_index()
    
    fig = px.bar(monthly_sales, x='month', y='sales', 
                 title="Sales by Month",
                 labels={'month': 'Month', 'sales': 'Total Sales ($)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("🎯 Key Insights")
    total_sales = sales_data['sales'].sum()
    avg_daily_sales = sales_data.groupby('date')['sales'].sum().mean()
    best_region = regional_sales.loc[regional_sales['Total Sales'].idxmax(), 'Region']
    best_product = product_sales.loc[product_sales['Total Sales'].idxmax(), 'Product']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sales", f"${total_sales:,.0f}")
    
    with col2:
        st.metric("Avg Daily Sales", f"${avg_daily_sales:,.0f}")
    
    with col3:
        st.metric("Top Region", best_region)
    
    with col4:
        st.metric("Top Product", best_product)

def customer_segmentation_project():
    """Customer Segmentation Project"""
    st.header("👥 Customer Segmentation Analysis")
    
    st.markdown("""
    **Project Overview:**
    Customer segmentation using demographic and behavioral data to identify distinct customer groups
    for targeted marketing strategies.
    
    **Technologies Used:** Python, Scikit-learn, K-Means Clustering, Statistical Analysis
    
    **Key Features:**
    - Demographic analysis
    - Behavioral segmentation
    - Customer profiling
    - Marketing recommendations
    """)
    
    _, customer_data, _ = create_sample_data()
    
    # Customer demographics
    st.subheader("👤 Customer Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(customer_data, x='age', title="Age Distribution",
                          nbins=20, labels={'age': 'Age', 'count': 'Number of Customers'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        gender_counts = customer_data['gender'].value_counts()
        fig = px.pie(values=gender_counts.values, names=gender_counts.index, 
                     title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Income vs Spending analysis
    st.subheader("💰 Income vs Spending Analysis")
    
    fig = px.scatter(customer_data, x='income', y='spending_score', 
                     color='gender', title="Income vs Spending Score",
                     labels={'income': 'Annual Income ($)', 'spending_score': 'Spending Score'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Age groups analysis
    st.subheader("👶👦👨👴 Age Group Analysis")
    
    customer_data['age_group'] = pd.cut(customer_data['age'], 
                                       bins=[0, 25, 35, 50, 100], 
                                       labels=['18-25', '26-35', '36-50', '50+'])
    
    age_group_stats = customer_data.groupby('age_group').agg({
        'income': 'mean',
        'spending_score': 'mean'
    }).reset_index()
    
    fig = px.bar(age_group_stats, x='age_group', y=['income', 'spending_score'],
                 title="Average Income and Spending by Age Group", barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer segments
    st.subheader("🎯 Customer Segments")
    
    # Create segments based on spending and income
    customer_data['segment'] = 'Low Value'
    customer_data.loc[(customer_data['income'] > customer_data['income'].median()) & 
                     (customer_data['spending_score'] > customer_data['spending_score'].median()), 'segment'] = 'High Value'
    customer_data.loc[(customer_data['income'] <= customer_data['income'].median()) & 
                     (customer_data['spending_score'] > customer_data['spending_score'].median()), 'segment'] = 'High Spender'
    customer_data.loc[(customer_data['income'] > customer_data['income'].median()) & 
                     (customer_data['spending_score'] <= customer_data['spending_score'].median()), 'segment'] = 'Conservative'
    
    segment_counts = customer_data['segment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(values=segment_counts.values, names=segment_counts.index, 
                     title="Customer Segment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        segment_stats = customer_data.groupby('segment').agg({
            'income': 'mean',
            'spending_score': 'mean',
            'age': 'mean'
        }).reset_index()
        
        st.write("**Segment Characteristics:**")
        st.dataframe(segment_stats.round(2), use_container_width=True)

def stock_prediction_project():
    """Stock Price Prediction Project"""
    st.header("📈 Stock Price Analysis & Prediction")
    
    st.markdown("""
    **Project Overview:**
    Time series analysis and prediction of stock prices using statistical methods and technical indicators.
    
    **Technologies Used:** Python, Pandas, Statistical Modeling, Technical Analysis
    
    **Key Features:**
    - Price trend analysis
    - Moving averages
    - Volatility analysis
    - Technical indicators
    """)
    
    _, _, stock_data = create_sample_data()
    
    # Calculate technical indicators
    stock_data['MA_7'] = stock_data['price'].rolling(window=7).mean()
    stock_data['MA_30'] = stock_data['price'].rolling(window=30).mean()
    stock_data['volatility'] = stock_data['price'].rolling(window=30).std()
    stock_data['daily_return'] = stock_data['price'].pct_change()
    
    # Price trends
    st.subheader("📊 Stock Price Trends")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['price'], 
                            name='Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['MA_7'], 
                            name='7-Day MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['MA_30'], 
                            name='30-Day MA', line=dict(color='green')))
    
    fig.update_layout(title="Stock Price with Moving Averages",
                     xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility analysis
    st.subheader("📉 Volatility Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(stock_data, x='date', y='volatility', 
                      title="30-Day Rolling Volatility")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(stock_data, x='daily_return', 
                          title="Daily Returns Distribution", nbins=50)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("📈 Performance Metrics")
    
    total_return = (stock_data['price'].iloc[-1] / stock_data['price'].iloc[0] - 1) * 100
    volatility_annual = stock_data['daily_return'].std() * np.sqrt(252) * 100
    max_drawdown = ((stock_data['price'] / stock_data['price'].cummax()) - 1).min() * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Return", f"{total_return:.2f}%")
    
    with col2:
        st.metric("Annual Volatility", f"{volatility_annual:.2f}%")
    
    with col3:
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")

def main():
    st.title("📊 Data Science Projects Portfolio")
    st.markdown("Showcase of various data science projects demonstrating different analytical techniques")
    
    # Project selection
    project_tabs = st.tabs(["📈 Sales Analysis", "👥 Customer Segmentation", "📊 Stock Analysis"])
    
    with project_tabs[0]:
        sales_analysis_project()
    
    with project_tabs[1]:
        customer_segmentation_project()
    
    with project_tabs[2]:
        stock_prediction_project()
    
    # Additional projects section
    st.markdown("---")
    st.subheader("🚀 Additional Project Ideas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Web Analytics Dashboard**
        - User behavior analysis
        - Conversion funnel optimization
        - A/B testing framework
        - Real-time monitoring
        """)
    
    with col2:
        st.markdown("""
        **🔍 Fraud Detection System**
        - Anomaly detection algorithms
        - Risk scoring models
        - Real-time alerting
        - Pattern recognition
        """)
    
    with col3:
        st.markdown("""
        **🤖 Recommendation Engine**
        - Collaborative filtering
        - Content-based recommendations
        - Hybrid approaches
        - Performance evaluation
        """)
    
    st.markdown("---")
    st.markdown("*These projects demonstrate various aspects of data science including EDA, visualization, machine learning, and statistical analysis. Each project showcases different tools and methodologies commonly used in the field.*")

if __name__ == "__main__":
    main()
