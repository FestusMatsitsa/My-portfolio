import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from scipy import stats
import io
import base64

st.set_page_config(page_title="Data Science Toolkit", page_icon="🔧", layout="wide")

# Custom CSS for dark mode
st.markdown("""
<style>
    .toolkit-section {
        background: linear-gradient(135deg, #262730 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 212, 170, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid rgba(0, 212, 170, 0.3);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.1);
    }
    .success-box {
        background-color: rgba(0, 212, 170, 0.1);
        border: 1px solid rgba(0, 212, 170, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #00d4aa;
    }
    .warning-box {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample dataset for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 12, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_years': np.random.normal(14, 3, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'loan_amount': np.random.normal(25000, 10000, n_samples)
    }
    
    # Create target variable
    data['loan_approved'] = (
        (data['credit_score'] > 600) & 
        (data['income'] > 30000) & 
        (data['age'] > 21)
    ).astype(int)
    
    # Add some categorical variables
    data['employment_type'] = np.random.choice(['Full-time', 'Part-time', 'Self-employed'], n_samples)
    data['region'] = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    
    return pd.DataFrame(data)

def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    st.subheader("📊 Exploratory Data Analysis")
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Overview:**")
        st.write(f"- Rows: {df.shape[0]:,}")
        st.write(f"- Columns: {df.shape[1]:,}")
        st.write(f"- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.write("**Data Types:**")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.write(f"- {dtype}: {count} columns")
    
    # Missing values analysis
    st.write("**Missing Values Analysis:**")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig = px.bar(x=missing_data.index, y=missing_data.values, 
                     title="Missing Values by Column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Descriptive statistics
    st.write("**Descriptive Statistics:**")
    st.dataframe(df.describe())
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.write("**Correlation Matrix:**")
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       labels=dict(x="Variables", y="Variables", color="Correlation"),
                       title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

def create_visualizations(df):
    """Create various data visualizations"""
    st.subheader("📈 Data Visualization")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Visualization options
    viz_type = st.selectbox("Select Visualization Type:", 
                           ["Distribution Plot", "Scatter Plot", "Box Plot", "Bar Chart", "Histogram"])
    
    if viz_type == "Distribution Plot" and len(numeric_cols) > 0:
        col = st.selectbox("Select Column:", numeric_cols)
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot" and len(numeric_cols) > 1:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols)
        with col2:
            y_col = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_col])
        
        color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
        
        if color_col == "None":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot" and len(numeric_cols) > 0:
        col = st.selectbox("Select Numeric Column:", numeric_cols)
        if len(categorical_cols) > 0:
            cat_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
            if cat_col != "None":
                fig = px.box(df, x=cat_col, y=col, title=f"{col} by {cat_col}")
            else:
                fig = px.box(df, y=col, title=f"Distribution of {col}")
        else:
            fig = px.box(df, y=col, title=f"Distribution of {col}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bar Chart" and len(categorical_cols) > 0:
        col = st.selectbox("Select Categorical Column:", categorical_cols)
        value_counts = df[col].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                     title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram" and len(numeric_cols) > 0:
        col = st.selectbox("Select Column:", numeric_cols)
        bins = st.slider("Number of Bins:", 10, 100, 30)
        fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)

def feature_engineering(df):
    """Perform feature engineering operations"""
    st.subheader("🔧 Feature Engineering")
    
    # Feature scaling
    st.write("**Feature Scaling:**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        scaling_method = st.selectbox("Select Scaling Method:", 
                                    ["None", "Standard Scaling", "Min-Max Scaling"])
        
        if scaling_method != "None":
            cols_to_scale = st.multiselect("Select columns to scale:", numeric_cols)
            
            if cols_to_scale:
                if scaling_method == "Standard Scaling":
                    scaler = StandardScaler()
                    df_scaled = df.copy()
                    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                    
                    st.success(f"✅ Applied Standard Scaling to {len(cols_to_scale)} columns")
                    st.write("**Before and After Scaling Comparison:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Original Data:")
                        st.dataframe(df[cols_to_scale].describe())
                    with col2:
                        st.write("Scaled Data:")
                        st.dataframe(df_scaled[cols_to_scale].describe())
    
    # Feature creation
    st.write("**Feature Creation:**")
    
    if 'age' in df.columns and 'income' in df.columns:
        if st.button("Create Age-Income Ratio Feature"):
            df['age_income_ratio'] = df['age'] / df['income'] * 1000
            st.success("✅ Created age_income_ratio feature")
            st.write("New feature statistics:")
            st.write(df['age_income_ratio'].describe())
    
    # Categorical encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) > 0:
        st.write("**Categorical Encoding:**")
        col_to_encode = st.selectbox("Select column to encode:", categorical_cols)
        encoding_method = st.selectbox("Select encoding method:", 
                                     ["Label Encoding", "One-Hot Encoding"])
        
        if st.button("Apply Encoding"):
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                df[f"{col_to_encode}_encoded"] = le.fit_transform(df[col_to_encode])
                st.success(f"✅ Applied Label Encoding to {col_to_encode}")
                
                # Show mapping
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                st.write("Encoding mapping:", mapping)
            
            elif encoding_method == "One-Hot Encoding":
                encoded_df = pd.get_dummies(df[col_to_encode], prefix=col_to_encode)
                st.success(f"✅ Applied One-Hot Encoding to {col_to_encode}")
                st.write(f"Created {encoded_df.shape[1]} new binary columns")
                st.dataframe(encoded_df.head())

def machine_learning_models(df):
    """Build and evaluate machine learning models"""
    st.subheader("🤖 Machine Learning Models")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for machine learning")
        return
    
    # Model type selection
    model_type = st.selectbox("Select Model Type:", 
                             ["Classification", "Regression", "Clustering"])
    
    if model_type in ["Classification", "Regression"]:
        # Target variable selection
        target_col = st.selectbox("Select Target Variable:", numeric_cols)
        feature_cols = st.multiselect("Select Feature Variables:", 
                                    [col for col in numeric_cols if col != target_col])
        
        if len(feature_cols) > 0:
            X = df[feature_cols].dropna()
            y = df[target_col].dropna()
            
            # Align X and y
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) > 0:
                # Train-test split
                test_size = st.slider("Test Size:", 0.1, 0.5, 0.2, 0.05)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42)
                
                if model_type == "Classification":
                    # Classification models
                    model_choice = st.selectbox("Select Model:", 
                                              ["Random Forest", "Logistic Regression"])
                    
                    if st.button("Train Model"):
                        if model_choice == "Random Forest":
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                        else:
                            model = LogisticRegression(random_state=42)
                        
                        # Convert to classification problem if needed
                        if len(np.unique(y)) > 10:
                            y_train_class = (y_train > y_train.median()).astype(int)
                            y_test_class = (y_test > y_test.median()).astype(int)
                        else:
                            y_train_class = y_train
                            y_test_class = y_test
                        
                        model.fit(X_train, y_train_class)
                        y_pred = model.predict(X_test)
                        
                        # Display results
                        accuracy = accuracy_score(y_test_class, y_pred)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.3f}")
                        with col2:
                            st.metric("Test Samples", len(y_test))
                        
                        # Feature importance
                        if hasattr(model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(importance_df, x='Importance', y='Feature', 
                                       orientation='h', title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif model_type == "Regression":
                    # Regression models
                    model_choice = st.selectbox("Select Model:", 
                                              ["Random Forest", "Linear Regression"])
                    
                    if st.button("Train Model"):
                        if model_choice == "Random Forest":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        else:
                            model = LinearRegression()
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Display results
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{r2:.3f}")
                        with col2:
                            st.metric("MSE", f"{mse:.3f}")
                        with col3:
                            st.metric("RMSE", f"{np.sqrt(mse):.3f}")
                        
                        # Prediction vs Actual plot
                        fig = px.scatter(x=y_test, y=y_pred, 
                                       labels={'x': 'Actual', 'y': 'Predicted'},
                                       title="Predicted vs Actual Values")
                        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                                    x1=y_test.max(), y1=y_test.max(),
                                    line=dict(dash="dash", color="red"))
                        st.plotly_chart(fig, use_container_width=True)
    
    elif model_type == "Clustering":
        # Clustering
        feature_cols = st.multiselect("Select Features for Clustering:", numeric_cols)
        
        if len(feature_cols) >= 2:
            n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
            
            if st.button("Perform Clustering"):
                X = df[feature_cols].dropna()
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Add clusters to dataframe
                df_clustered = X.copy()
                df_clustered['Cluster'] = clusters
                
                # Visualize clusters (2D)
                if len(feature_cols) >= 2:
                    fig = px.scatter(df_clustered, x=feature_cols[0], y=feature_cols[1], 
                                   color='Cluster', title="Clustering Results")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster statistics
                st.write("**Cluster Statistics:**")
                cluster_stats = df_clustered.groupby('Cluster').agg({
                    col: ['mean', 'std'] for col in feature_cols
                }).round(3)
                st.dataframe(cluster_stats)

def statistical_analysis(df):
    """Perform statistical analysis"""
    st.subheader("📈 Statistical Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for statistical analysis")
        return
    
    # Statistical tests
    test_type = st.selectbox("Select Statistical Test:", 
                           ["Correlation Test", "T-Test", "Chi-Square Test", "Normality Test"])
    
    if test_type == "Correlation Test":
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Variable 1:", numeric_cols)
        with col2:
            var2 = st.selectbox("Variable 2:", [col for col in numeric_cols if col != var1])
        
        if st.button("Run Correlation Test"):
            # Remove NaN values
            data_clean = df[[var1, var2]].dropna()
            
            if len(data_clean) > 2:
                correlation, p_value = stats.pearsonr(data_clean[var1], data_clean[var2])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Correlation Coefficient", f"{correlation:.4f}")
                with col2:
                    st.metric("P-value", f"{p_value:.4f}")
                
                # Interpretation
                if p_value < 0.05:
                    st.success(f"✅ Significant correlation found (p < 0.05)")
                else:
                    st.warning(f"⚠️ No significant correlation (p >= 0.05)")
                
                # Scatter plot with regression line
                fig = px.scatter(data_clean, x=var1, y=var2, 
                               trendline="ols", title=f"Correlation: {var1} vs {var2}")
                st.plotly_chart(fig, use_container_width=True)
    
    elif test_type == "Normality Test":
        col = st.selectbox("Select Column:", numeric_cols)
        
        if st.button("Run Normality Test"):
            data = df[col].dropna()
            
            if len(data) > 3:
                # Shapiro-Wilk test
                statistic, p_value = stats.shapiro(data[:5000])  # Limit sample size
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Statistic", f"{statistic:.4f}")
                with col2:
                    st.metric("P-value", f"{p_value:.4f}")
                
                # Interpretation
                if p_value < 0.05:
                    st.warning(f"⚠️ Data is not normally distributed (p < 0.05)")
                else:
                    st.success(f"✅ Data appears to be normally distributed (p >= 0.05)")
                
                # Q-Q plot
                fig = go.Figure()
                stats.probplot(data, dist="norm", plot=None)
                theoretical_quantiles, sample_quantiles = stats.probplot(data, dist="norm", plot=None)
                
                fig.add_trace(go.Scatter(x=theoretical_quantiles[0], y=theoretical_quantiles[1],
                                       mode='markers', name='Data points'))
                fig.add_trace(go.Scatter(x=theoretical_quantiles[0], y=theoretical_quantiles[0],
                                       mode='lines', name='Normal distribution line'))
                fig.update_layout(title="Q-Q Plot", xaxis_title="Theoretical Quantiles",
                                yaxis_title="Sample Quantiles")
                st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🔧 Data Science Toolkit")
    st.markdown("---")
    
    st.markdown("""
    Welcome to the comprehensive Data Science Toolkit! This interactive tool provides:
    - **Data Upload & Preview** - Load and explore your datasets
    - **Exploratory Data Analysis** - Comprehensive statistical analysis
    - **Data Visualization** - Create interactive charts and plots
    - **Feature Engineering** - Transform and prepare your data
    - **Machine Learning** - Build and evaluate models
    - **Statistical Analysis** - Perform hypothesis testing and more
    """)
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Toolkit Navigation")
    toolkit_section = st.sidebar.selectbox("Select Tool:", [
        "Data Upload & Preview",
        "Exploratory Data Analysis", 
        "Data Visualization",
        "Feature Engineering",
        "Machine Learning",
        "Statistical Analysis"
    ])
    
    # Data upload section
    st.header("📁 Data Upload & Management")
    
    data_source = st.radio("Select Data Source:", 
                          ["Upload CSV File", "Use Sample Dataset", "Enter Data Manually"])
    
    df = None
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    elif data_source == "Use Sample Dataset":
        if st.button("Generate Sample Dataset"):
            df = generate_sample_data()
            st.success("✅ Sample dataset generated successfully!")
            st.info("This is a synthetic dataset for loan approval prediction with features like age, income, credit score, etc.")
    
    elif data_source == "Enter Data Manually":
        st.info("Feature coming soon! Use CSV upload or sample dataset for now.")
    
    # Display data preview
    if df is not None:
        st.subheader("📊 Data Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{df.shape[1]:,}")
        with col3:
            st.metric("Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("Categorical Cols", len(df.select_dtypes(include=['object']).columns))
        
        # Show data sample
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10))
        
        # Column information
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info)
        
        # Tool sections
        st.markdown("---")
        
        if toolkit_section == "Data Upload & Preview":
            st.info("✅ Data loaded successfully! Select other tools from the sidebar to analyze your data.")
        
        elif toolkit_section == "Exploratory Data Analysis":
            perform_eda(df)
        
        elif toolkit_section == "Data Visualization":
            create_visualizations(df)
        
        elif toolkit_section == "Feature Engineering":
            feature_engineering(df)
        
        elif toolkit_section == "Machine Learning":
            machine_learning_models(df)
        
        elif toolkit_section == "Statistical Analysis":
            statistical_analysis(df)
    
    else:
        st.info("👆 Please upload a CSV file or generate a sample dataset to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>🔧 Data Science Toolkit by Festus Matsitsa Bombo</p>
    <p>Built with Streamlit, Python, and various data science libraries</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
