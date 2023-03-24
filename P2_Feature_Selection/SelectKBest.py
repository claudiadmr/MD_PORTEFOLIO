# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:24:41 2023

@author: catia
"""

import numpy as np

class SelectKBest:
    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k
    
    def fit(self, X, y):
        scores, p_values = self.score_func(X, y)
        self.scores_ = scores
        self.p_values_ = p_values
        
    def transform(self, X):
        mask = np.argsort(self.p_values_)[:self.k]
        return X[:, mask]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    
from scipy.stats import f_regression

# create a sample dataset
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([1, 2, 3])

# create a SelectKBest object and fit_transform the data
selector = SelectKBest(score_func=f_regression, k=2)
X_new = selector.fit_transform(X, y)

# print the selected features
print(X_new)