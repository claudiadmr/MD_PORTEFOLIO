# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:41:24 2023

@author: catia
"""

import numpy as np
from scipy.stats import f_oneway


def f_classif(dataset):
    """
    Performs one-way ANOVA for feature selection.
    Parameters
    ----------
    dataset : Dataset
        The dataset to perform ANOVA on.
    Returns
    -------
    tuple
        A tuple containing the F-value and p-value for each feature.
    """

    # Group samples by class
    groups = [dataset.X[dataset.y == i] for i in np.unique(dataset.y)]

    # Perform ANOVA for each feature
    F_scores, p_values = [], []
    for i in range(dataset.X.shape[1]):
        f, p = f_oneway(*[group[:, i] for group in groups])
        F_scores.append(f)
        p_values.append(p)

    return F_scores, p_values



# Create a toy dataset with 3 features and 4 samples
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([0, 0, 1, 1])
class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

dataset = Dataset(X, y)

# Call the f_classif function
F_scores, p_values = f_classif(dataset)

# Print the F-scores and p-values
print("F-scores:", F_scores)
print("p-values:", p_values)





#%%
