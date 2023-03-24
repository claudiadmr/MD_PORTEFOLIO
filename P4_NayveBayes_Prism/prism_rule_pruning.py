import numpy as np

def prism_rule_pruning(X, y, delta):
    # Initialize the set of candidate rules with all possible binary attributes
    n_samples, n_features = X.shape
    candidate_rules = np.eye(n_features, dtype=bool)
    
    while True:
        # Compute the accuracy of each candidate rule
        accuracies = np.zeros(n_features)
        for i in range(n_features):
            if candidate_rules[i].any():
                # Evaluate the accuracy of the rule that uses the i-th feature as the condition
                condition = X[:, i]
                rule = condition == 1
                accuracies[i] = np.mean(y[rule])
        
        # Find the rule with the highest accuracy and remove it if it's not statistically significant
        best_rule = np.argmax(accuracies)
        if accuracies[best_rule] < delta:
            break
        else:
            candidate_rules[best_rule] = False
    
    # Apply the remaining rules to predict the class labels
    predictions = np.ones(n_samples)
    for i in range(n_features):
        if candidate_rules[i].any():
            condition = X[:, i]
            rule = condition == 1
            predictions[rule] = y[rule].mean()
    
    return predictions


X = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 1]
])
y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 1])

# Apply PRISM rule pruning
predictions = prism_rule_pruning(X, y, delta=0.8)

# Print the predicted class labels and the accuracy
print("Predicted class labels:", predictions)
print("Accuracy:", np.mean(predictions == y))


