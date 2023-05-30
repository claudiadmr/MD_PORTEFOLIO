import numpy as np
import unittest
from f_classif import f_classif

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


class FClassifTest(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        y = np.array([0, 1, 0, 1])
        self.dataset = Dataset(X,y)
     

    def test_f_classif(self):
        # Call f_classif
        F_scores, p_values = f_classif(self.dataset)

        # Assert the expected values
        expected_F_scores = [0.5, 0.5, 0.5]
        expected_p_values = [0.552786404500042, 0.552786404500042, 0.552786404500042]
        self.assertListEqual(F_scores, expected_F_scores)
        self.assertListEqual(p_values, expected_p_values)

if __name__ == "__main__":
    unittest.main()