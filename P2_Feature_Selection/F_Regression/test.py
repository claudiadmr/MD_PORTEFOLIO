import unittest
import numpy as np
import scipy.stats
from f_regression import f_regression


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


class FRegressionTest(unittest.TestCase):
    def test_f_regression(self):
        # Create a sample dataset
        np.random.seed(42)
        X = np.random.randn(1000, 3)
        y = X[:, 0] + 2 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(1000)
        dataset = Dataset(X, y)

        # Calculate F-value and p-value
        F, p_value = f_regression(dataset)

        # Perform assertions to check the results
        self.assertAlmostEqual(F, 1570.479834294179, places=6)
        self.assertAlmostEqual(p_value, 1.1102230246251565e-16, places=6)


if __name__ == "__main__":
    unittest.main()
