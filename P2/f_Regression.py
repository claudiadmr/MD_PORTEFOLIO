# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:03:10 2023

@author: catia
"""

import numpy as np
from scipy import stats


def f_regression(dataset):
    """
    Performs linear regression for feature selection using numpy.
    Parameters
    ----------
    dataset : Dataset
        The dataset to perform linear regression on.
    Returns
    -------
    tuple
        A tuple containing the F-value and p-value for each feature.
    """

    # Split the data into features (X) and target (y)
    X = dataset.X
    y = dataset.y

    # Add a column of ones to X for the intercept term
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    # Compute the coefficients using linear regression
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Compute the predicted values and residuals
    y_pred = np.dot(X, beta)
    residuals = y - y_pred

    # Compute the total sum of squares
    SST = np.sum((y - np.mean(y)) ** 2)

    # Compute the residual sum of squares
    SSR = np.sum(residuals ** 2)

    # Compute the degrees of freedom
    # Compute the degrees of freedom
    n = X.shape[0]
    p = dataset.X.shape[1]  # number of features in the dataset
    df_reg = p
    df_res = n - p - 1

    # Compute the F-value and p-value
    F = (SST - SSR) / df_reg / (SSR / df_res)
    p_value = 1 - stats.f.cdf(F, df_reg, df_res)

    return F, p_value


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


# create a sample dataset with 1000 rows and 3 columns
np.random.seed(42)
X = np.random.randn(1000, 3)
y = X[:, 0] + 2 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(1000)
dataset = Dataset(X, y)

# call the function
F, p_value = f_regression(dataset)

# print the results
print("F-values:", F)
print("p-values:", p_value)

