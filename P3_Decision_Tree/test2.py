import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTreeClassifier
from decision_tree import build_tree
from decision_tree import prune_tree
from decision_tree import entropy
from decision_tree import gain_ratio
from decision_tree import gini
from decision_tree import pre_pruning


class DecisionTreeClassifierTest(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset for testing
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def test_fit_predict(self):
        # Test the fit and predict methods
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertEqual(y_pred.shape[0], self.y_test.shape[0])

    def test_accuracy_score(self):
        # Test the accuracy_score function
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_entropy(self):
        # Test the entropy function
        y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0])
        expected_entropy = 0.9910760598382222
        Entropy = entropy(y)
        self.assertAlmostEqual(Entropy, expected_entropy, places=6)

    def test_gini(self):
        # Test the gini function
        y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0])
        expected_gini = 0.49382716049382713
        Gini = gini(y)
        self.assertAlmostEqual(Gini, expected_gini, places=6)

    def test_build_tree(self):
        # Test the build_tree function
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        feature_indices = np.array([0, 1])
        max_depth = 2
        min_samples_leaf = 1
        impurity_measure = "entropy"
        max_leaf_size = None
        p_value = None
        tree = build_tree(X, y, feature_indices, max_depth, min_samples_leaf, impurity_measure, max_leaf_size, p_value)
        self.assertIsNotNone(tree)

    def test_prune_tree(self):
        # Test the prune_tree function
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        feature_indices = np.array([0, 1])
        max_depth = 2
        min_samples_leaf = 1
        impurity_measure = "entropy"
        max_leaf_size = None
        p_value = None
        tree = build_tree(X, y, feature_indices, max_depth, min_samples_leaf, impurity_measure, max_leaf_size, p_value)
        prune_method = "your_prune_method"  # Provide a valid pruning method
        pruned_tree = prune_tree(tree, X, y, prune_method)
        self.assertIsNotNone(pruned_tree)

    def test_pre_pruning(self):
        # Test the pre_pruning function
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        max_depth = 2
        min_samples_leaf = 1
        p_value = None
        pruned = pre_pruning(X, y, max_depth, min_samples_leaf, p_value)
        self.assertFalse(pruned)


if __name__ == '__main__':
    unittest.main()
