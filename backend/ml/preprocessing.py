import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_new = X.copy()
        X_new['claim_frequency'] = X_new['previous_claims'] / (X_new['policy_tenure'] + 0.1)
        X_new['coverage_to_premium'] = X_new['coverage_amount'] / (X_new['premium'] + 1)
        X_new['premium_burden'] = X_new['premium'] / (X_new['income'] + 1)
        X_new['customer_stability'] = (X_new['payment_history'] * X_new['renewal_count']) / (X_new['policy_tenure'] + 1)
        X_new['age_risk'] = X_new['age'] / 75.0
        return X_new

class InsurancePreprocessor:
    def __init__(self):
        self.num_cols = ['age', 'income', 'policy_tenure', 'premium', 'coverage_amount',
                         'vehicle_age', 'num_dependents', 'previous_claims', 'payment_history',
                         'renewal_count', 'health_score', 'claim_frequency', 'coverage_to_premium',
                         'premium_burden', 'customer_stability', 'age_risk']
        self.cat_cols = ['gender', 'occupation', 'location', 'policy_type']
        
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('cat', cat_pipeline, self.cat_cols)
        ])
        
        self.pipeline = Pipeline([
            ('engineer', FeatureEngineer()),
            ('prep', self.preprocessor)
        ])
        
    def fit_transform(self, X_train):
        out = self.pipeline.fit_transform(X_train)
        return out, self.get_feature_names()
        
    def transform(self, X_test):
        return self.pipeline.transform(X_test)
        
    def get_feature_names(self):
        # Extract feature names from ColumnTransformer
        cat_encoder = self.pipeline.named_steps['prep'].named_transformers_['cat'].named_steps['encoder']
        cat_features = cat_encoder.get_feature_names_out(self.cat_cols).tolist()
        return self.num_cols + cat_features
        
    def detect_outliers(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        counts = {}
        for col in numeric_cols:
            if col in ['claim_filed', 'risk_score_base']: continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            
            mean = df[col].mean()
            std = df[col].std()
            z_outliers = ((df[col] - mean).abs() > 3 * std).sum() if std > 0 else 0
            
            counts[col] = {'iqr_outliers': int(iqr_outliers), 'z_outliers': int(z_outliers)}
        return counts
