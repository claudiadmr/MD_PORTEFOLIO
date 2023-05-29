# Importing the necessary library
import numpy as np


# Define the NaiveBayes class
class NaiveBayes:
    def __init__(self):
        # Initialize parameters list and classes
        self.parameters = []
        self.classes = None

    # Define the fit method to train the classifier
    def fit(self, X, y):
        # Save feature matrix and target vector as instance variables
        self.X = X
        self.y = y

        # Identify the unique classes in the target vector
        self.classes = np.unique(y)

        # Calculate the mean and variance for each feature for each class
        for c in self.classes:
            # Select rows of X where the class is c
            X_c = X[y == c]

            # Add a new list to parameters to store the parameters for class c
            self.parameters.append([])

            # Calculate mean and variance for each column in the selected rows
            for col in range(X.shape[1]):
                mean = np.mean(X_c[:, col])
                var = np.var(X_c[:, col])
                # Save the mean and variance for this column in parameters
                self.parameters[-1].append((mean, var))

    # Define the predict method to make predictions
    def predict(self, X):
        # Initialize the probabilities with zeroes
        probabilities = np.zeros((X.shape[0], len(self.classes)))

        # For each class, calculate the probability that each instance belongs to this class
        for i, (c, params) in enumerate(zip(self.classes, self.parameters)):
            # Prior probability P(y) with Laplace smoothing
            probabilities[:, i] = np.mean(self.y == c) + 1e-9

            # For each column, multiply the probability by the probability density function of the Gaussian distribution
            for col, (mean, var) in enumerate(params):
                probabilities[:, i] *= (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((X[:, col] - mean) ** 2 / (2 * var)))

        # For each instance, return the class with the highest probability
        return self.classes[np.argmax(probabilities, axis=1)]


# Define a function to perform one-hot encoding
def one_hot_encoding(X):
    # Define the encoding mapping
    encoding_map = {
        'Sunny': 0,
        'Overcast': 1,
        'Rain': 2,
        'Hot': 3,
        'Mild': 4,
        'Cool': 5,
        'High': 6,
        'Normal': 7,
        'Weak': 8,
        'Strong': 9
    }

    # Initialize encoded matrix
    encoded = np.zeros((X.shape[0], X.shape[1]), dtype=int)

    # Encode the input matrix
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            encoded[i, j] = encoding_map.get(X[i, j], -1)  # Get the encoded value from the map; if not found, set as -1

    return encoded


# Load the Tennis dataset
tennis_data = np.array([
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

# Split the dataset into features and labels
X = tennis_data[:, :-1]
y = tennis_data[:, -1]

# Encode the features using one-hot encoding
X_encoded = one_hot_encoding(X)

# Define test data
X_test = np.array([
    ['Sunny', 'Hot', 'High', 'Weak'],
    ['Sunny', 'Cool', 'High', 'Strong'],
    ['Rain', 'Cool', 'Normal', 'Weak']
])

# Encode the test data using one-hot encoding
X_test_encoded = one_hot_encoding(X_test)

# Train the Naive Bayes classifier
nb = NaiveBayes()
nb.fit(X_encoded, y)

# Make predictions using the trained model
y_pred = nb.predict(X_test_encoded)

# Print the predictions
print('Predictions:', y_pred)
