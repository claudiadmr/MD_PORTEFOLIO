import numpy as np


class VarianceThreshold:
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.
    Parameters
    ----------
    threshold: float
        The threshold value to use for feature selection. Features with a
        training-set variance lower than this threshold will be removed.
    Attributes
    ----------
    variance: array-like, shape (n_features,)
        The variance of each feature.
    """
    def __init__(self, threshold=0.0):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.
        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        self.threshold = threshold
        self.variance = None

    def fit(self, X):
        """
        Fit the VarianceThreshold model according to the given training data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        Returns
        -------
        self : object
        """
        self.variance = np.var(X, axis=0)
        return self

    def transform(self, X):
        """
        It removes all features whose variance does not meet the threshold.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input data.
        Returns
        -------
        X_selected: array-like, shape (n_samples, n_features_new)
            The input data with selected features.
        """
        features_mask = self.variance > self.threshold
        X_selected = X[:, features_mask]
        if X_selected is not None:
            print('variance does not meet the threshold in:')
            print(X_selected)
        else:
            print('variance meets the threshold')
        return X_selected

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input data.
        Returns
        -------
        X_selected: array-like, shape (n_samples, n_features_new)
            The input data with selected features.
        """
        self.fit(X)
        self.transform(X)


def test_VarianceThreshold():
    # Create a toy dataset
    X = np.array([[1, 2, 1, 3],
                  [5, 1, 4, 3],
                  [0, 1, 1, 3]])
    y = np.array([0, 1, 0])

    # Create a VarianceThreshold object
    selector = VarianceThreshold(threshold= 1)

    # Fit the selector to the data
    selector.fit_transform(X)
    
test_VarianceThreshold()


