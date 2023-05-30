import unittest
import numpy as np

from mlp import MLP


class TestMLPMethods(unittest.TestCase):

    def setUp(self):
        # Setting up a simple MLP instance before each test
        X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
        y = np.array([[0], [1], [1], [0]])
        self.mlp = MLP(X, y, hidden_size=4)

    def test_feedforward(self):
        self.mlp.feedforward()
        # Check that the output is of correct shape
        self.assertEqual(self.mlp.output.shape, self.mlp.y.shape)

    def test_backprop(self):
        # Save weights before backprop
        weights1_before = self.mlp.weights1.copy()
        weights2_before = self.mlp.weights2.copy()
        self.mlp.feedforward()
        self.mlp.backprop(learning_rate=0.5)

        # Assert that weights have changed after backprop
        self.assertFalse(np.array_equal(weights1_before, self.mlp.weights1))
        self.assertFalse(np.array_equal(weights2_before, self.mlp.weights2))

    def test_predict(self):
        prediction = self.mlp.predict(np.array([1, 0, 1]))
        # Assert that predict returns a numpy array of shape (1,)
        self.assertEqual(prediction.shape, (1,))

    def test_costFunction(self):
        cost = self.mlp.costFunction()
        # Assert that cost is a scalar value (float)
        self.assertIsInstance(cost, float)

    def test_buildModel(self):
        # Assert that no error is raised during training
        try:
            self.mlp.buildModel(learning_rate=0.5, epochs=10)
        except Exception as e:
            self.fail("buildModel raised exception: " + str(e))


# Running the tests
if __name__ == '__main__':
    unittest.main()
