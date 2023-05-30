from SelectKBest import SelectKBest
from sklearn.feature_selection import f_regression

import numpy as np

def test_select_k_best():
    # Create a sample dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])

    # Create an instance of SelectKBest and specify the score function and k value
    selector = SelectKBest(score_func=f_regression, k=2)

    # Fit and transform the dataset
    X_new = selector.fit_transform(X, y)

    # Expected result
    expected_X_new = np.array([[1,2],[4,5],[7,8]])

    # Check if the selected features match the expected result
    if np.array_equal(X_new, expected_X_new):
        print("Test passed!")
    else:
        print("Test failed!")
        print("X_new:", X_new)
        print("Expected X_new:", expected_X_new)


if __name__ == "__main__":
    # Run the test
    test_select_k_best()