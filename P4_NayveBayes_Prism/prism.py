import numpy as np


class Prism:
    # Constructor: Initializes the rules list
    def __init__(self):
        # List of rules, each rule is a tuple of form (feature_index, feature_value, class_label)
        self.rules = []

    # fit function: Trains the model on given data
    def fit(self, X, y):
        # X: np.ndarray, input features
        # y: np.ndarray, target labels

        # Until there are instances left
        while X.shape[0] > 0:
            # Find the best rule that covers the most instances
            best_rule = self.find_best_rule(X, y)
            if best_rule is None:
                break
            # Add the best rule to the rules list
            self.rules.append(best_rule)

            # Remove the instances covered by the best rule
            X, y = self.remove_covered(X, y, best_rule)

    # find_best_rule function: Finds the rule that covers the most instances
    def find_best_rule(self, X, y):
        # X: np.ndarray, input features
        # y: np.ndarray, target labels
        # Returns: tuple, (feature_index, feature_value, class_label)

        best_rule = None
        best_accuracy = -1

        # For each feature
        for feature_index in range(X.shape[1]):
            # For each unique value of the feature
            for feature_value in np.unique(X[:, feature_index]):
                # For each unique class label
                for class_label in np.unique(y):
                    rule = (feature_index, feature_value, class_label)
                    # Evaluate the accuracy of the rule
                    accuracy = self.evaluate_rule(X, y, rule)
                    # If the accuracy is higher than the best accuracy found so far
                    if accuracy > best_accuracy:
                        best_rule = rule
                        best_accuracy = accuracy

        # Return the best rule
        return best_rule

    # evaluate_rule function: Evaluates the accuracy of a rule
    def evaluate_rule(self, X, y, rule):
        # X: np.ndarray, input features
        # y: np.ndarray, target labels
        # rule: tuple, (feature_index, feature_value, class_label)
        # Returns: float, accuracy of the rule

        # Unpack the rule into feature index, feature value and class label
        feature_index, feature_value, class_label = rule

        # Find the instances that match the rule
        matches_rule = X[:, feature_index] == feature_value

        # Return the accuracy of the rule, i.e., the proportion of instances that match the rule and belong to the correct class
        return np.mean(y[matches_rule] == class_label)

    # remove_covered function: Removes the instances covered by a rule
    def remove_covered(self, X, y, rule):
        # X: np.ndarray, input features
        # y: np.ndarray, target labels
        # rule: tuple, (feature_index, feature_value, class_label)
        # Returns: tuple, (np.ndarray, np.ndarray), instances and labels not covered by the rule

        # Unpack the rule into feature index and feature value
        feature_index, feature_value, _ = rule

        # Find the instances not covered by the rule
        uncovered = X[:, feature_index] != feature_value

        # Return the uncovered instances and their labels
        return X[uncovered], y[uncovered]

    # predict function: Predicts the class labels of given instances
    def predict(self, X):
        # X: np.ndarray, input features
        # Returns: np.ndarray, predicted labels

        # Initialize an array for predictions
        predictions = np.empty(X.shape[0])

        # For each instance
        for i, instance in enumerate(X):
            # For each rule
            for rule in self.rules:
                # Unpack the rule into feature index, feature value and class label
                feature_index, feature_value, class_label = rule
                # If the instance matches the rule
                if instance[feature_index] == feature_value:
                    # Predict the class label according to the rule
                    predictions[i] = class_label
                    # Stop checking further rules
                    break

        # Return the predictions
        return predictions

    # __repr__ function: Returns a string representation of the model
    def __repr__(self):
        # Returns a string containing the list of rules
        return "Prism classifier with rules: " + str(self.rules)


def test():
    # Test the PRISM algorithm
    # Input features
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    # Target labels
    y = np.array([1, 1, 0, 0])

    # Create a PRISM classifier
    prism = Prism()

    # Train the classifier
    prism.fit(X, y)

    # Print the classifier
    print(prism)

    # Predict the labels of the training instances
    print("predictions: ", prism.predict(X))


if __name__ == "__main__":
    # Run the test
    test()
