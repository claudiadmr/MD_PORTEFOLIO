from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Node:
    def __init__(self, split_feature=None, split_threshold=None, left_child=None, right_child=None, label=None,
                 impurity=None):
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.left_child = left_child
        self.right_child = right_child
        self.label = label
        self.impurity = impurity


def entropy(y_pred):
    _, counts = np.unique(y_pred, return_counts=True)
    probs = counts / len(y_pred)
    return -np.sum(probs * np.log2(probs))


def gini( y_pred):
    _, counts = np.unique(y_pred, return_counts=True)
    probs = counts / len(y_pred)
    return 1 - np.sum(probs ** 2)


def gain_ratio(X, y, feature_indices):
    base_impurity = entropy(y)
    max_gain_ratio = 0
    best_feature_index = None
    best_threshold = None
    for feature_index in feature_indices:
        values = np.unique(X[:, feature_index])
        for threshold in values:
            left_indices = np.where(X[:, feature_index] <= threshold)[0]
            right_indices = np.where(X[:, feature_index] > threshold)[0]
            left_ratio = len(left_indices) / len(X)
            right_ratio = len(right_indices) / len(X)
            split_impurity = left_ratio * entropy(y[left_indices]) + right_ratio * entropy(y[right_indices])
            information_gain = base_impurity - split_impurity
            split_info = -left_ratio * np.log2(left_ratio) - right_ratio * np.log2(right_ratio)
            gain_r = information_gain / split_info if split_info != 0 else 0
            if gain_r > max_gain_ratio:
                max_gain_ratio = gain_r
                best_feature_index = feature_index
                best_threshold = threshold
    return (best_feature_index, best_threshold) if max_gain_ratio > 0 else None


def select_best_split_entropy(X, y, feature_indices):
    best_feature_index, best_threshold, best_info_gain = None, None, -float("inf")
    H_y = entropy(y)

    for feature_index in feature_indices:
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = np.where(X[:, feature_index] <= threshold)[0].astype(int)
            right_indices = np.where(X[:, feature_index] > threshold)[0].astype(int)
            H_y_x = (len(left_indices) / len(y)) * entropy(y[left_indices]) + \
                    (len(right_indices) / len(y)) * entropy(y[right_indices])
            info_gain = H_y - H_y_x
            if info_gain > best_info_gain:
                best_feature_index = feature_index
                best_threshold = threshold
                best_info_gain = info_gain

    return best_feature_index, best_threshold


def select_best_split_gini_index(X, y, feature_indices):
    best_feature_index, best_threshold, best_gini = None, None, np.inf
    for feature_index in feature_indices:
        for threshold in np.unique(X[:, feature_index]):
            left_indices = np.where(X[:, feature_index] <= threshold)[0]
            right_indices = np.where(X[:, feature_index] > threshold)[0]
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            current_gini = (len(left_indices) / len(y)) * gini(y[left_indices]) \
                           + (len(right_indices) / len(y)) * gini(y[right_indices])
            if current_gini < best_gini:
                best_feature_index, best_threshold, best_gini = feature_index, threshold, current_gini
    return best_feature_index, best_threshold


def split_information(n_left, n_right):
    n_total = n_left + n_right
    p_left = n_left / n_total
    p_right = n_right / n_total
    if p_left == 0 or p_right == 0:
        return 0
    else:
        return - p_left * np.log2(p_left) - p_right * np.log2(p_right)


def compute_gain_ratio(y, left_indices, right_indices):
    # Compute the entropy of the whole dataset
    entropy_before_split = entropy(y)

    # Compute the weighted average entropy of the left and right partitions
    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])
    n_instances = len(y)
    n_left = len(left_indices)
    n_right = n_instances - n_left
    weighted_avg_entropy = (n_left / n_instances) * left_entropy + (n_right / n_instances) * right_entropy

    # Compute the split information
    split_info = split_information(n_left, n_right)

    # Compute the gain ratio
    gain_ratio = (entropy_before_split - weighted_avg_entropy) / split_info if split_info != 0 else 0

    return gain_ratio


def select_best_split_gain_ratio(X, y, feature_indices):
    best_feature_index, best_threshold, max_gain_ratio = None, None, 0.0
    for feature_index in feature_indices:
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = np.where(X[:, feature_index] <= threshold)[0]
            right_indices = np.where(X[:, feature_index] > threshold)[0]
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            gain_ratio = compute_gain_ratio(y, left_indices, right_indices)
            if gain_ratio > max_gain_ratio:
                best_feature_index, best_threshold, max_gain_ratio = feature_index, threshold, gain_ratio
    return best_feature_index, best_threshold


def build_tree(X, y, feature_indices, max_depth=None,
               min_samples_leaf=1, impurity_measure="entropy"):
    if len(np.unique(y)) == 1:
        return Node(label=y[0])
    if max_depth is not None and max_depth == 0:
        return Node(label=np.argmax(np.bincount(y)))
    if len(X) < min_samples_leaf:
        return Node(label=np.argmax(np.bincount(y)))
    if impurity_measure == "entropy":
        impurity_func = entropy
        best_criteria_func = select_best_split_entropy
    elif impurity_measure == "gini_index":
        impurity_func = gini
        best_criteria_func = select_best_split_gini_index
    elif impurity_measure == "gain_ratio":
        impurity_func = entropy
        best_criteria_func = select_best_split_gain_ratio
    else:
        raise ValueError("Invalid impurity measure specified")
    best_feature_index, best_threshold = best_criteria_func(X, y, feature_indices)
    if best_feature_index is None:
        return Node(label=np.argmax(np.bincount(y)))
    left_indices = np.where(X[:, best_feature_index] <= best_threshold)[0].astype(int)
    right_indices = np.where(X[:, best_feature_index] > best_threshold)[0].astype(int)
    left_child = build_tree(X[left_indices], y[left_indices], feature_indices,
                            max_depth - 1 if max_depth is not None else None, min_samples_leaf, impurity_measure)
    right_child = build_tree(X[right_indices], y[right_indices], feature_indices,
                             max_depth - 1 if max_depth is not None else None, min_samples_leaf, impurity_measure)
    return Node(split_feature=best_feature_index, split_threshold=best_threshold, left_child=left_child,
                right_child=right_child, impurity=impurity_func(y))


def prune_tree(node, X, y, prune_method, threshold=None):
    if node.label is not None:
        return node
    node.left_child = prune_tree(node.left_child, X, y, prune_method, threshold)
    node.right_child = prune_tree(node.right_child, X, y, prune_method, threshold)
    left_label = None
    right_label = None
    if node.left_child.label is not None and node.right_child.label is not None:
        if prune_method == "majority_voting":
            left_label = node.left_child.label
            right_label = node.right_child.label
            if np.sum(y == left_label) > np.sum(y == right_label):
                node.left_child = None
                node.right_child = None
                node.label = left_label
            else:
                node.left_child = None
                node.right_child = None
                node.label = right_label
    elif prune_method == "class_threshold":
        node_samples = len(y)
        if node.left_child.label is not None:
            left_label = node.left_child.label
            left_samples = len(np.where(y[node.left_child.indices] == left_label)[0])
        else:
            left_samples = 0
        if node.right_child.label is not None:
            right_label = node.right_child.label
            right_samples = len(np.where(y[node.right_child.indices] == right_label)[0])
        else:
            right_samples = 0
        if (left_samples / node_samples >= threshold) and (right_samples / node_samples >= threshold):
            node.left_child = None
            node.right_child = None
        elif node.left_child.label is not None and node.right_child.label is not None:
            if np.sum(y == left_label) > np.sum(y == right_label):
                node.label = left_label
            else:
                node.label = right_label
    return node


def predict_sample(x, node):
    if node.label is not None:
        return node.label
    if x[node.split_feature] <= node.split_threshold:
        return predict_sample(x, node.left_child)
    else:
        return predict_sample(x, node.right_child)


def predict(X, node):
    return np.array([predict_sample(x, node) for x in X])


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_leaf=1, impurity_measure="entropy", attribute_selection="gain_ratio",
                 prune_method=None, threshold=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.impurity_measure = impurity_measure
        self.attribute_selection = attribute_selection
        self.prune_method = prune_method
        self.threshold = threshold

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature_indices_ = np.arange(self.n_features_)
        if self.attribute_selection == "random":
            self.feature_indices_ = np.random.choice(self.feature_indices_, size=int(np.sqrt(self.n_features_)),
                                                     replace=False)
        self.root_ = build_tree(X, y, self.feature_indices_, self.max_depth, self.min_samples_leaf,
                                self.impurity_measure)
        if self.prune_method is not None:
            self.root_ = prune_tree(self.root_, X, y, self.prune_method, self.threshold)
        return self

    def predict(self, X):
        return predict(X, self.root_)


# Carrega o conjunto de dados iris
df = pd.read_csv("tennis.csv")
label_enc = LabelEncoder()
df['play'] = label_enc.fit_transform(df['play'])
X = df.drop("play", axis=1).values
y = df["play"].values

# Divide o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria uma instância do modelo DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, impurity_measure="entropy",
                             attribute_selection="gain_ratio", prune_method="reduced_error", threshold=0.1)

# Treina o modelo no conjunto de treinamento
dtc.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = dtc.predict(X_test)

# Calcula a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)

print("Precisão do modelo:", accuracy)
