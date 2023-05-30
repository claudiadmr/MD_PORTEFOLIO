import unittest
import numpy as np

from VarianceThreshold import VarianceThreshold

class VarianceThresholdTest(unittest.TestCase):
    def test_transform(self):
        # Create a sample dataset
        X = np.array([[1, 2, 1, 3],[5, 1, 4, 3],[0, 1, 1, 3]]) 
        threshold = 2.0

        # Create a VarianceThreshold object with the given threshold
        selector = VarianceThreshold(threshold)

        # Fit and transform the dataset
        X_selected = selector.fit_transform(X)

        # Define the expected output
        expected_output = np.array([[1], [5],[0]])

        # Assert the selected features match the expected output
        self.assertTrue(np.array_equal(X_selected, expected_output))

    def test_transform_no_features_selected(self):
        # Create a sample dataset with zero variance features
        X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        threshold = 2.0

        # Create a VarianceThreshold object with the given threshold
        selector = VarianceThreshold(threshold)

        # Fit and transform the dataset
        X_selected = selector.fit_transform(X)

        # Assert that no features are selected
        self.assertEqual(X_selected.shape[1], 0)

if __name__ == '__main__':
    unittest.main()