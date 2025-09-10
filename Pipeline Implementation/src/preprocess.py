import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class CorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9, target=None):
        self.threshold = threshold
        self.target = target
        self.to_drop_ = []

    def fit(self, X, y=None):
        if y is not None and isinstance(X, pd.DataFrame):
            corr_with_target = {}
            for col in X.columns:
                if np.issubdtype(X[col].dtype, np.number):
                    corr, _ = pearsonr(X[col], y)
                    corr_with_target[col] = abs(corr)
            # drop columns with almost no correlation
            self.to_drop_ = [col for col, c in corr_with_target.items() if c < 0.05]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.to_drop_, errors="ignore")
        return X

class ChiSquareSelector(BaseEstimator, TransformerMixin):
    def __init__(self, p_value_threshold=0.05):
        self.p_value_threshold = p_value_threshold
        self.to_keep_ = []

    def fit(self, X, y=None):
        if y is not None and isinstance(X, pd.DataFrame):
            for col in X.columns:
                if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
                    # encode categories as integers
                    encoded = LabelEncoder().fit_transform(X[col].astype(str))
                    chi2_val, p = chi2(encoded.reshape(-1, 1), y)
                    if p < self.p_value_threshold:
                        self.to_keep_.append(col)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.to_keep_] if self.to_keep_ else pd.DataFrame(index=X.index)
        return X

def build_preprocessor(numeric_cols, categorical_cols, y=None):
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('corr_selector', CorrelationSelector(target=y))  # drop weak numeric features
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('chi_selector', ChiSquareSelector()),           # keep only dependent cats
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ])
    return preprocessor
