import pandas as pd
import numpy as np

class NaiveBayes:
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = []
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.parameters.append([])
            
            for j in range(X.shape[1]):
                mean = np.mean(X_c[:, j])
                var = np.var(X_c[:, j])
                self.parameters[i].append((mean, var))
        
    def predict(self, X):
        probabilities = np.zeros((X.shape[0], len(self.classes)))
        
        for i, c in enumerate(self.classes):
            prior = np.sum(self.y == c) / self.y.shape[0]
            probabilities[:,i] = prior
            
            for j in range(X.shape[1]):
                mean, var = self.parameters[i][j]
                p = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((X[:,j] - mean)**2 / (2 * var)))
                probabilities[:,i] *= p
        
        return self.classes[np.argmax(probabilities, axis=1)]


# load the Tennis dataset
tennis_data = np.array([    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
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

# print the shape of the dataset
print('Tennis dataset shape:', tennis_data.shape)

# split the dataset into features and labels
X = tennis_data[:, :-1]
y = tennis_data[:, -1]

# encode the features using one-hot encoding
X_encoded = np.zeros((X.shape[0], X.shape[1]), dtype=int)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if X[i, j] == 'Sunny':
            X_encoded[i, j] = 0
        elif X[i, j] == 'Overcast':
            X_encoded[i, j] = 1
        elif X[i, j] == 'Rain':
            X_encoded[i, j] = 2
        elif X[i, j] == 'Hot':
            X_encoded[i, j] = 3
        elif X[i, j] == 'Mild':
            X_encoded[i, j] = 4
        elif X[i, j] == 'Cool':
            X_encoded[i, j] = 5
        elif X[i, j] == 'High':
            X_encoded[i, j] = 6
        elif X[i, j] == 'Normal':
            X_encoded[i, j] = 7
        elif X[i, j] == 'Weak':
            X_encoded[i, j] = 8
        elif X[i, j] == 'Strong':
            X_encoded[i, j] = 9

# train the Naive Bayes classifier
nb = NaiveBayes()
nb.fit(X_encoded, y)

# make predictions
X_test = np.array([
    ['Sunny', 'Hot', 'High', 'Weak'],
    ['Sunny', 'Cool', 'High', 'Strong'],
    ['Rain', 'Cool', 'Normal', 'Weak']
])

X_test_encoded = np.zeros((X_test.shape[0], X_test.shape[1]), dtype=int)                          
# encode the test features using one-hot encoding
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        if X_test[i, j] == 'Sunny':
            X_test_encoded[i, j] = 0
        elif X_test[i, j] == 'Overcast':
            X_test_encoded[i, j] = 1
        elif X_test[i, j] == 'Rain':
            X_test_encoded[i, j] = 2
        elif X_test[i, j] == 'Hot':
            X_test_encoded[i, j] = 3
        elif X_test[i, j] == 'Mild':
            X_test_encoded[i, j] = 4
        elif X_test[i, j] == 'Cool':
            X_test_encoded[i, j] = 5
        elif X_test[i, j] == 'High':
            X_test_encoded[i, j] = 6
        elif X_test[i, j] == 'Normal':
            X_test_encoded[i, j] = 7
        elif X_test[i, j] == 'Weak':
            X_test_encoded[i, j] = 8
        elif X_test[i, j] == 'Strong':
            X_test_encoded[i, j] = 9

# make predictions
y_pred = nb.predict(X_test_encoded)

# print the predictions
print('Predictions:', y_pred)


'''
this  code implements the Naive Bayes algorithm for classification. The fit method fits the model to the training data,
and the predict method predicts the class labels for new data.

The fit method takes two arguments, X and y, which are the training data and corresponding labels, respectively. It first
computes the set of unique classes in the label y, and stores them in the self.classes attribute. It then computes the mean
and variance of each feature in each class, and stores them in the self.parameters attribute. The mean and variance are 
computed using the training data X for each feature, separately for each class.

The predict method takes one argument, X, which is the data to be classified. It first initializes an array called probabilities
with zeros, with dimensions (X.shape[0], len(self.classes)). This array will store the probability of each class for each data
point. For each class c, it computes the prior probability of that class as the proportion of training data points with label c,
and stores it in prior. It then computes the probability of each feature given the class, assuming a normal distribution with mean
and variance given by the self.parameters attribute. It multiplies these probabilities together to obtain the joint probability of
the features given the class, and multiplies this by the prior probability to obtain the overall probability of the class given 
the data. It stores this probability in the appropriate entry of the probabilities array.

Finally, the predict method returns the predicted class for each data point, which is the class with the highest probability according
to the probabilities array, using the argmax function. The class labels are returned in the order given by self.classes.
'''
