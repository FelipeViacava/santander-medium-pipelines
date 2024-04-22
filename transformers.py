import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        return X_.drop(self.columns, axis=1)
    
class InsertNa(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, value=0):
        self.columns = columns
        self.value = value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_[self.columns] = X_[self.columns].replace(self.value, np.nan)
        return X_
    
class CountNa(BaseEstimator, TransformerMixin):
    
    def __init__(self, prefix, columns):
        self.columns = columns
        self.prefix = prefix
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_[f"{self.prefix}_na_count"] = X_[self.columns].isna().sum(axis=1)
        return X_
    
class SumCols(BaseEstimator, TransformerMixin):
    
    def __init__(self, prefix, columns):
        self.columns = columns
        self.prefix = prefix
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_[f"{self.prefix}_sum"] = X_[self.columns].sum(axis=1)
        return X_