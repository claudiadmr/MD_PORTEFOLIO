import unittest
import numpy as np
from decision_tree import DecisionTreeClassifier

class DecisionTreeClassifierTest(unittest.TestCase):
    def test_fit(self):
        # Test the fit method of DecisionTreeClassifier
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, impurity_measure="entropy",
                                     attribute_selection="gain_ratio")
        clf.fit(X, y)
        
        # Assert that the classifier has been fitted
        self.assertIsNotNone(clf.root_)
        
        # Assert that the number of features has been calculated
        self.assertEqual(clf.n_features_, X.shape[1])
        
        # Assert that the feature indices have been initialized
        self.assertTrue(np.array_equal(clf.feature_indices_, np.arange(clf.n_features_)))
        
    def test_predict(self):
        # Test the predict method of DecisionTreeClassifier
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, impurity_measure="entropy",
                                     attribute_selection="gain_ratio")
        clf.fit(X, y)
        
        # Make predictions on the training data
        y_pred = clf.predict(X)
        
        # Assert that the predicted labels have the same shape as the input data
        self.assertEqual(y_pred.shape, y.shape)
        
        # Assert that the predicted labels are in the correct range
        self.assertTrue(np.all(np.logical_or(y_pred == 0, y_pred == 1)))

if __name__ == '__main__':
    unittest.main()
