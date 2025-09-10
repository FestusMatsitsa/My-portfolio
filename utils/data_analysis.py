import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """
    Comprehensive data analysis utility class
    """
    
    def __init__(self, df):
        """Initialize with a pandas DataFrame"""
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    def basic_info(self):
        """Return basic information about the dataset"""
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': dict(self.df.dtypes)
        }
        return info
    
    def missing_value_analysis(self):
        """Analyze missing values in the dataset"""
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        return missing_df
    
    def outlier_detection(self, method='iqr'):
        """Detect outliers using IQR or Z-score method"""
        outliers = {}
        
        for col in self.numeric_columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outlier_mask = z_scores > 3
            
            outliers[col] = {
                'count': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(self.df)) * 100,
                'indices': self.df[outlier_mask].index.tolist()
            }
        
        return outliers
    
    def correlation_analysis(self, threshold=0.5):
        """Analyze correlations between numeric variables"""
        if len(self.numeric_columns) < 2:
            return None
        
        corr_matrix = self.df[self.numeric_columns].corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs
        }
    
    def statistical_summary(self):
        """Generate comprehensive statistical summary"""
        summary = {}
        
        # Numeric variables
        if self.numeric_columns:
            numeric_summary = self.df[self.numeric_columns].describe()
            
            # Add additional statistics
            for col in self.numeric_columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    numeric_summary.loc['skewness', col] = stats.skew(data)
                    numeric_summary.loc['kurtosis', col] = stats.kurtosis(data)
                    numeric_summary.loc['median', col] = data.median()
                    numeric_summary.loc['mode', col] = data.mode().iloc[0] if len(data.mode()) > 0 else np.nan
            
            summary['numeric'] = numeric_summary
        
        # Categorical variables
        if self.categorical_columns:
            categorical_summary = {}
            for col in self.categorical_columns:
                value_counts = self.df[col].value_counts()
                categorical_summary[col] = {
                    'unique_count': self.df[col].nunique(),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'value_counts': dict(value_counts.head(10))
                }
            
            summary['categorical'] = categorical_summary
        
        return summary
    
    def feature_engineering_suggestions(self):
        """Suggest feature engineering operations"""
        suggestions = []
        
        # Check for potential date columns
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(self.df[col].head(100))
                    suggestions.append(f"Column '{col}' might be a date - consider converting to datetime")
                except:
                    pass
        
        # Check for highly skewed numeric columns
        for col in self.numeric_columns:
            data = self.df[col].dropna()
            if len(data) > 0:
                skewness = abs(stats.skew(data))
                if skewness > 2:
                    suggestions.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f}) - consider log transformation")
        
        # Check for columns with many unique values (potential for binning)
        for col in self.numeric_columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.9:
                suggestions.append(f"Column '{col}' has many unique values - consider binning")
        
        # Check for categorical columns with many categories
        for col in self.categorical_columns:
            unique_count = self.df[col].nunique()
            if unique_count > 20:
                suggestions.append(f"Column '{col}' has {unique_count} categories - consider grouping rare categories")
        
        return suggestions
    
    def data_quality_report(self):
        """Generate comprehensive data quality report"""
        report = {
            'basic_info': self.basic_info(),
            'missing_values': self.missing_value_analysis(),
            'outliers': self.outlier_detection(),
            'correlations': self.correlation_analysis(),
            'statistical_summary': self.statistical_summary(),
            'feature_suggestions': self.feature_engineering_suggestions()
        }
        
        return report

class FeatureEngineer:
    """
    Feature engineering utility class
    """
    
    def __init__(self, df):
        self.df = df.copy()
    
    def handle_missing_values(self, strategy='mean', columns=None):
        """Handle missing values with various strategies"""
        if columns is None:
            columns = self.df.columns
        
        df_processed = self.df.copy()
        
        for col in columns:
            if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_processed[col].fillna(df_processed[col].median(), inplace=True)
                    elif strategy == 'mode':
                        df_processed[col].fillna(df_processed[col].mode().iloc[0], inplace=True)
                    elif strategy == 'forward_fill':
                        df_processed[col].fillna(method='ffill', inplace=True)
                    elif strategy == 'backward_fill':
                        df_processed[col].fillna(method='bfill', inplace=True)
                else:
                    # Categorical columns
                    if strategy == 'mode':
                        df_processed[col].fillna(df_processed[col].mode().iloc[0], inplace=True)
                    elif strategy == 'unknown':
                        df_processed[col].fillna('Unknown', inplace=True)
        
        return df_processed
    
    def scale_features(self, columns=None, method='standard'):
        """Scale numeric features"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        df_scaled = self.df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        
        return df_scaled, scaler
    
    def encode_categorical(self, columns=None, method='label'):
        """Encode categorical variables"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        
        df_encoded = self.df.copy()
        encoders = {}
        
        for col in columns:
            if col in df_encoded.columns:
                if method == 'label':
                    le = LabelEncoder()
                    df_encoded[f"{col}_encoded"] = le.fit_transform(df_encoded[col].astype(str))
                    encoders[col] = le
                elif method == 'onehot':
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    encoders[col] = list(dummies.columns)
        
        return df_encoded, encoders
    
    def create_polynomial_features(self, columns, degree=2):
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(self.df[columns])
        
        feature_names = poly.get_feature_names_out(columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=self.df.index)
        
        return poly_df, poly
    
    def create_interaction_features(self, column_pairs):
        """Create interaction features between specified column pairs"""
        df_interactions = self.df.copy()
        
        for col1, col2 in column_pairs:
            if col1 in df_interactions.columns and col2 in df_interactions.columns:
                if (df_interactions[col1].dtype in ['int64', 'float64'] and 
                    df_interactions[col2].dtype in ['int64', 'float64']):
                    
                    df_interactions[f"{col1}_{col2}_interaction"] = (
                        df_interactions[col1] * df_interactions[col2]
                    )
        
        return df_interactions
    
    def bin_numeric_features(self, column, bins=5, labels=None):
        """Bin numeric features into categories"""
        df_binned = self.df.copy()
        
        if column in df_binned.columns and df_binned[column].dtype in ['int64', 'float64']:
            df_binned[f"{column}_binned"] = pd.cut(
                df_binned[column], 
                bins=bins, 
                labels=labels,
                include_lowest=True
            )
        
        return df_binned

def perform_feature_selection(X, y, k=10, score_func=None):
    """Perform feature selection using univariate statistical tests"""
    if score_func is None:
        # Determine if it's classification or regression
        if len(np.unique(y)) < 20:  # Assume classification
            score_func = f_classif
        else:  # Assume regression
            score_func = f_regression
    
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = selector.scores_[selector.get_support()]
    
    return X_selected, selected_features, feature_scores
