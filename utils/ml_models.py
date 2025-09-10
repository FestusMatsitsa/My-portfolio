import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, adjusted_rand_score
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class MLModelBuilder:
    """
    Comprehensive machine learning model builder and evaluator
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scalers = {}
        
        # Define available models
        self.classification_models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        self.regression_models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
        
        self.clustering_models = {
            'KMeans': KMeans(random_state=42),
            'DBSCAN': DBSCAN(),
            'Agglomerative': AgglomerativeClustering()
        }
    
    def prepare_data(self, df, target_column, feature_columns=None, test_size=0.2, scale_features=True):
        """Prepare data for machine learning"""
        if feature_columns is None:
            feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col != target_column]
        
        # Remove rows with missing target values
        df_clean = df.dropna(subset=[target_column])
        
        # Prepare features and target
        X = df_clean[feature_columns].copy()
        y = df_clean[target_column].copy()
        
        # Handle missing values in features
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if self._is_classification(y) else None
        )
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            self.scalers['feature_scaler'] = scaler
        
        return X_train, X_test, y_train, y_test
    
    def _is_classification(self, y):
        """Determine if the problem is classification or regression"""
        unique_values = len(np.unique(y))
        total_values = len(y)
        
        # If less than 20 unique values or if unique values are less than 5% of total
        return unique_values < 20 or (unique_values / total_values) < 0.05
    
    def train_classification_models(self, X_train, X_test, y_train, y_test, models_to_train=None):
        """Train multiple classification models"""
        if models_to_train is None:
            models_to_train = list(self.classification_models.keys())
        
        results = {}
        
        for model_name in models_to_train:
            if model_name in self.classification_models:
                print(f"Training {model_name}...")
                
                model = self.classification_models[model_name]
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                self.models[model_name] = model
        
        self.results['classification'] = results
        return results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test, models_to_train=None):
        """Train multiple regression models"""
        if models_to_train is None:
            models_to_train = list(self.regression_models.keys())
        
        results = {}
        
        for model_name in models_to_train:
            if model_name in self.regression_models:
                print(f"Training {model_name}...")
                
                model = self.regression_models[model_name]
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2,
                    'rmse': rmse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                self.models[model_name] = model
        
        self.results['regression'] = results
        return results
    
    def perform_clustering(self, X, n_clusters_range=None, algorithms=None):
        """Perform clustering analysis"""
        if algorithms is None:
            algorithms = ['KMeans', 'DBSCAN']
        
        if n_clusters_range is None:
            n_clusters_range = range(2, 8)
        
        results = {}
        
        # Standardize features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for algorithm in algorithms:
            if algorithm == 'KMeans':
                # Try different numbers of clusters
                for n_clusters in n_clusters_range:
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = model.fit_predict(X_scaled)
                    
                    # Calculate silhouette score
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                    
                    results[f'KMeans_{n_clusters}'] = {
                        'model': model,
                        'labels': cluster_labels,
                        'n_clusters': n_clusters,
                        'silhouette_score': silhouette_avg,
                        'inertia': model.inertia_
                    }
            
            elif algorithm == 'DBSCAN':
                model = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = model.fit_predict(X_scaled)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
                if n_clusters > 1:
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                else:
                    silhouette_avg = -1  # Invalid clustering
                
                results['DBSCAN'] = {
                    'model': model,
                    'labels': cluster_labels,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette_avg,
                    'n_noise_points': list(cluster_labels).count(-1)
                }
        
        self.results['clustering'] = results
        return results
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid, cv=5):
        """Perform hyperparameter tuning using GridSearchCV"""
        if model_name in self.classification_models:
            base_model = self.classification_models[model_name]
            scoring = 'accuracy'
        elif model_name in self.regression_models:
            base_model = self.regression_models[model_name]
            scoring = 'r2'
        else:
            raise ValueError(f"Model {model_name} not found")
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
    
    def feature_importance_analysis(self, model_name, feature_names):
        """Analyze feature importance for tree-based models"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return None
    
    def model_comparison(self, problem_type='classification'):
        """Compare performance of different models"""
        if problem_type not in self.results:
            return None
        
        results = self.results[problem_type]
        
        if problem_type == 'classification':
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[model]['accuracy'] for model in results.keys()],
                'Precision': [results[model]['precision'] for model in results.keys()],
                'Recall': [results[model]['recall'] for model in results.keys()],
                'F1_Score': [results[model]['f1_score'] for model in results.keys()],
                'CV_Mean': [results[model]['cv_mean'] for model in results.keys()],
                'CV_Std': [results[model]['cv_std'] for model in results.keys()]
            }).sort_values('Accuracy', ascending=False)
        
        elif problem_type == 'regression':
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'R2_Score': [results[model]['r2_score'] for model in results.keys()],
                'MSE': [results[model]['mse'] for model in results.keys()],
                'MAE': [results[model]['mae'] for model in results.keys()],
                'RMSE': [results[model]['rmse'] for model in results.keys()],
                'CV_Mean': [results[model]['cv_mean'] for model in results.keys()],
                'CV_Std': [results[model]['cv_std'] for model in results.keys()]
            }).sort_values('R2_Score', ascending=False)
        
        return comparison_df
    
    def predict_new_data(self, model_name, X_new):
        """Make predictions on new data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        model = self.models[model_name]
        
        # Scale the new data if scaler exists
        if 'feature_scaler' in self.scalers:
            X_new_scaled = self.scalers['feature_scaler'].transform(X_new)
            X_new = pd.DataFrame(X_new_scaled, columns=X_new.columns, index=X_new.index)
        
        predictions = model.predict(X_new)
        
        # Get probabilities for classification models
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }

class ModelEvaluator:
    """
    Model evaluation and visualization utilities
    """
    
    @staticmethod
    def plot_learning_curve(model, X, y, cv=5):
        """Generate learning curve data"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )
        
        return {
            'train_sizes': train_sizes,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }
    
    @staticmethod
    def calculate_prediction_intervals(y_true, y_pred):
        """Calculate prediction intervals for regression"""
        residuals = y_true - y_pred
        std_residuals = np.std(residuals)
        
        # 95% prediction interval
        lower_bound = y_pred - 1.96 * std_residuals
        upper_bound = y_pred + 1.96 * std_residuals
        
        return lower_bound, upper_bound
    
    @staticmethod
    def model_diagnostics(y_true, y_pred, model_type='regression'):
        """Generate model diagnostic information"""
        diagnostics = {}
        
        if model_type == 'regression':
            residuals = y_true - y_pred
            
            diagnostics['residuals'] = residuals
            diagnostics['residual_mean'] = np.mean(residuals)
            diagnostics['residual_std'] = np.std(residuals)
            diagnostics['residual_skewness'] = pd.Series(residuals).skew()
            diagnostics['residual_kurtosis'] = pd.Series(residuals).kurtosis()
            
            # Normality test for residuals
            from scipy.stats import shapiro
            if len(residuals) <= 5000:
                stat, p_value = shapiro(residuals)
                diagnostics['normality_test'] = {'statistic': stat, 'p_value': p_value}
        
        elif model_type == 'classification':
            diagnostics['unique_predictions'] = len(np.unique(y_pred))
            diagnostics['prediction_distribution'] = pd.Series(y_pred).value_counts().to_dict()
        
        return diagnostics

# Utility functions for common parameter grids
def get_common_param_grids():
    """Return common parameter grids for hyperparameter tuning"""
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 1]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    }
    
    return param_grids
