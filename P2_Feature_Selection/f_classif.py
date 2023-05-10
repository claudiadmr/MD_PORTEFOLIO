import numpy as np
from scipy.stats import f_oneway


def f_classif(dataset):
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

'''
This function takes a dataset and performs ANOVA (analysis of variance) for each feature in the dataset.
It groups samples by class, and then calculates F-scores and p-values for each feature. The F-scores represent
the ratio of between-group variability to within-group variability, and the p-values represent the probability
that the null hypothesis (that the means of all groups are equal) is true.

The function returns two lists: F_scores, which contains the F-scores for each feature, and p_values, which 
contains the p-values for each feature.
'''
