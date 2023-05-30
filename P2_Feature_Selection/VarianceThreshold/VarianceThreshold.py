import numpy as np

class VarianceThreshold:
    def __init__(self, threshold=0.0):
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        self.threshold = threshold
        self.variance = None
        
    def fit(self, X):
        self.variance = np.var(X, axis=0)
        return self

    def transform(self, X):
        features_mask = self.variance > self.threshold
        X_selected = X[:, features_mask]
        if X_selected.shape[1] == 0:
            print("No features meet the variance threshold.")
        else:
            print("Selected Features:")
            print(X_selected)
        return X_selected

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def test1():
   # Create a sample dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    threshold = 2.0

    # Create a VarianceThreshold object with the given threshold
    selector = VarianceThreshold(threshold)

    # Fit and transform the dataset
    # print the selected features
    print("Test2")
    X_selected = selector.fit_transform(X)


def test2():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # create a VarianceThreshold object with threshold=0
    threshold = 6.0
    selector = VarianceThreshold(threshold=threshold)

    # fit_transform the data
    # print the selected features
    print("Test2")
    X_new = selector.fit_transform(X)

    
   
if __name__ == "__main__":
    # Run the test
    test1()
    test2()



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
