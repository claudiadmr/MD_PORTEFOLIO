import unittest
import numpy as np
from dataset import Dataset


class DatasetTestCase(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        self.dataset = Dataset()
        self.dataset.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.dataset.y = np.array([10, 11, 12])
        self.dataset.features = ['feat1', 'feat2', 'feat3']
        self.dataset.label = 'label'

    def test_get_X(self):
        # Test the get_X() method
        self.assertTrue(np.array_equal(self.dataset.get_X(), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))

    def test_set_X(self):
        # Test the set_X() method
        new_X = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        self.dataset.set_X(new_X)
        self.assertTrue(np.array_equal(self.dataset.X, new_X))

    def test_get_y(self):
        # Test the get_y() method
        self.assertTrue(np.array_equal(self.dataset.get_y(), np.array([10, 11, 12])))

    def test_get_features(self):
        # Test the get_features() method
        self.assertEqual(self.dataset.get_features(), ['feat1', 'feat2', 'feat3'])

    def test_get_label(self):
        # Test the get_label() method
        self.assertEqual(self.dataset.get_label(), 'label')

    def test_read_csv(self):
        # Test the read_csv() method
        self.dataset.read_csv('wine.csv')
        self.assertTrue(np.array_equal(self.dataset.X, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
        self.assertTrue(np.array_equal(self.dataset.y, np.array([10, 11, 12])))
        self.assertEqual(self.dataset.features, ['feat1', 'feat2', 'feat3'])
        self.assertEqual(self.dataset.label, 'label')

    def test_write_csv(self):
        # Test the write_csv() method
        self.dataset.write_csv('output.csv')

    def test_replace_to_null(self):
        # Test the replace_to_null() method
        self.dataset.replace_to_null(2)
        expected_X = np.array([[1, np.nan, 3], [4, 5, 6], [7, 8, 9]])
        expected_y = np.array([10, 11, 12])
        self.assertTrue(np.array_equal(self.dataset.X, expected_X))
        self.assertTrue(np.array_equal(self.dataset.y, expected_y))

    

if __name__ == '__main__':
    unittest.main()



