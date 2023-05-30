import numpy as np


# Sigmoid function used for activation function
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# Derivative of the sigmoid function, used during backpropagation
def sigmoid_derivative(x):
    return x * (1.0 - x)


# MLP Class
class MLP:
    # Constructor of MLP class
    def __init__(self, x, y, hidden_size):
        self.input = x  # Input data
        self.weights1 = np.random.rand(self.input.shape[1], hidden_size)  # Weights for input layer
        self.weights2 = np.random.rand(hidden_size, 1)  # Weights for hidden layer
        self.y = y  # Output data
        self.output = np.zeros(y.shape)  # Current output of the network

    # Forward propagation through the network
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))  # First layer after activation
        self.output = sigmoid(np.dot(self.layer1, self.weights2))  # Output layer after activation

    # Backpropagation through the network (used for learning)
    def backprop(self, learning_rate):
        # Application of the chain rule to compute derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                 self.weights2.T) * sigmoid_derivative(self.layer1))

        # Update weights with the derivative of the loss function
        self.weights1 += learning_rate * d_weights1
        self.weights2 += learning_rate * d_weights2

    # Function to predict the output for a given input
    def predict(self, x):
        self.input = x
        self.feedforward()  # Generate output
        return self.output  # Return generated output

    # Cost function of the network
    def costFunction(self):
        self.feedforward()  # Generate output
        # Return the sum of square difference between the network output and actual output
        return np.sum((self.y - self.output) ** 2)

    # Training function of the network
    # Method for building (training) the model
    def buildModel(self, learning_rate=0.01, epochs=1000):
        # The for loop will repeat 'epochs' number of times.
        # Each iteration represents one full pass (forward and backward) over the entire dataset which is one epoch.
        for epoch in range(epochs):
            # First, the forward pass - the model applies weights to the inputs and runs through the network layers
            self.feedforward()

            # Second, the backward pass (Backpropagation) - the model adjusts its weights based on the error from the forward pass
            self.backprop(learning_rate)

            # For every 100th epoch (i.e., every time the remainder of epoch/100 is zero)
            if epoch % 100 == 0:
                # Print the current epoch number and the cost function value
                # This provides a visibility into how the training is going - the cost should be going down over time
                print("Epoch: {} - Cost: {}".format(epoch, self.costFunction()))


def test():
    # Dados de entrada
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    # Dados de saída
    y = np.array([[0], [1], [1], [0]])

    mlp = MLP(X, y, hidden_size=4)

    # Construir modelo
    mlp.buildModel(learning_rate=0.5, epochs=1500)

    # Prever saída para um exemplo
    print(mlp.predict(np.array([1, 0, 1])))


if __name__ == '__main__':
    test()
