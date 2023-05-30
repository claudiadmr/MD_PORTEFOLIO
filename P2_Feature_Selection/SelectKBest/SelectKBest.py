import numpy as np
from sklearn.feature_selection import f_regression

class SelectKBest:
    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k
        
    #Define method for estimating the F and p values for each feature
    def fit(self, X, y):
        scores, p_values = self.score_func(X, y)
        self.scores_ = scores
        self.p_values_ = p_values
        
    #Define method for selecting the k features with the lowest p-value
    def transform(self, X):
        mask = np.argsort(self.p_values_)[:self.k]
        return X[:, mask]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
def test1():
    # Create a sample dataset
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    y = np.array([1, 2, 3])

    # create a SelectKBest object and fit_transform the data
    selector = SelectKBest(score_func=f_regression, k=3)
    X_new = selector.fit_transform(X, y)

    # print the selected features
    print("test1")
    print(X_new)

def test2():
    # Create a sample dataset
    X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    y = np.array([1, 2, 3])

    # create a SelectKBest object and fit_transform the data
    selector = SelectKBest(score_func=f_regression, k=3)
    X_new = selector.fit_transform(X, y)

    # print the selected features
    print("test2")
    print(X_new)

if __name__ == "__main__":
    # Run the test
    test1()
    test2()


'''
This is a basic implementation of the SelectKBest class that can be used for feature selection.

The fit method takes two input arrays X and y, where X is the feature matrix and y is the target array.
The method calculates the scores and p-values for each feature using the score_func function and stores
them in the scores_ and p_values_ attributes.

The transform method takes an input feature matrix X and returns the subset of features with the lowest
p-values. It uses the argsort method to sort the p-values in ascending order and returns the indices of
the first k elements of the sorted array. It then uses these indices to select the corresponding columns
from the input matrix X and returns the resulting subset of features.

The fit_transform method is a convenience method that calls fit and transform in sequence and returns the
resulting subset of features.
'''
