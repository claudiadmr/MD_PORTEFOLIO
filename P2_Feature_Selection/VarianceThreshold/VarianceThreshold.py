import numpy as np


class VarianceThreshold:
    # Define method for initializing the class with a threshold parameter
    def __init__(self, threshold=0.0):
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        self.threshold = threshold
        self.variance = None
        
    # Define method for computing the variance of each feature
    def fit(self, X):
        self.variance = np.var(X, axis=0)
        return self

    #Define method for applying the variance thresholding on the input data
    def transform(self, X):
        features_mask = self.variance > self.threshold
        X_selected = X[:, features_mask]
        if X_selected is not None:
            print('variance does not meet the threshold in:')
            print(X_selected)
        else:
            print('variance meets the threshold')
        return X_selected

    def fit_transform(self, X):
        self.fit(X)
        self.transform(X)



'''
This is a Python class that implements a feature selection method based on variance thresholding.

The class has three methods:

- __init__ method initializes the class with a threshold parameter that is set to a default value of 0.0.
If the value of threshold is less than 0, a ValueError is raised. The self.threshold and self.variance 
variables are also initialized.

- fit method computes the variance of each feature in the input data X along the 0-th axis, which corresponds
to the feature axis. The computed variance is stored in the self.variance variable.

- transform method applies the variance thresholding on the input data X. It selects the features whose variance
is greater than the threshold value and returns the corresponding subset of X. If none of the features meet the 
threshold, a message is printed to the console. The selected features are returned.
'''
