# Import the necessary libraries
import unittest
import numpy as np
from NaiveBayes import NaiveBayes, one_hot_encoding  # replace with your actual module


# Define a class for the tests
class TestNaiveBayes(unittest.TestCase):

    # Define the setup method
    def setUp(self):
        self.nb = NaiveBayes()
        self.tennis_data = np.array([
            ['Sunny', 'Hot', 'High', 'Weak', 'No'],
            ['Sunny', 'Hot', 'High', 'Strong', 'No'],
            ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
            ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
            ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
            ['Sunny', 'Mild', 'High', 'Weak', 'No'],
            ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
            ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
            ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
            ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'High', 'Strong', 'No']
        ])
        self.X = self.tennis_data[:, :-1]
        self.y = self.tennis_data[:, -1]

    # Test the fit method
    def test_fit(self):
        self.nb.fit(one_hot_encoding(self.X), self.y)
        self.assertIsNotNone(self.nb.parameters)
        self.assertIsNotNone(self.nb.classes)

    # Test the predict method
    def test_predict(self):
        self.nb.fit(one_hot_encoding(self.X), self.y)
        X_test = np.array([['Rain', 'Cool', 'Normal', 'Weak']])
        y_pred = self.nb.predict(one_hot_encoding(X_test))
        self.assertIsNotNone(y_pred)

    # Test one_hot_encoding function
    def test_one_hot_encoding(self):
        encoded = one_hot_encoding(self.X)
        self.assertEqual(encoded.shape, self.X.shape)


# Run the tests
if __name__ == '__main__':
    unittest.main()
