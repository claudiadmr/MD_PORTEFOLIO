# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:47:57 2023

@author: catia
"""

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

