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

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


def test():
    # Create a sample dataset
    X = np.array([[2, 6, 8], [1, 5, 2], [4, 2, 1], [8, 11, 10]])
    y = np.array([0, 1, 0, 1])
    dataset = Dataset(X, y)
    # Call f_classif
    F_scores, p_values = f_classif(dataset)

    # Print the results
    print("F-scores:", F_scores)
    print("p-values:", p_values)


if __name__ == "__main__":
    # Run the test
    test()



'''
This function takes a dataset and performs ANOVA (analysis of variance) for each feature in the dataset.
It groups samples by class, and then calculates F-scores and p-values for each feature. The F-scores represent
the ratio of between-group variability to within-group variability, and the p-values represent the probability
that the null hypothesis (that the means of all groups are equal) is true.

The function returns two lists: F_scores, which contains the F-scores for each feature, and p_values, which 
contains the p-values for each feature.
'''
