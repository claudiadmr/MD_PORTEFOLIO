import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DecisionTreeClassifier:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, pre_pruning=None,
                 post_pruning=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.pre_pruning = pre_pruning
        self.post_pruning = post_pruning
        self.random_state = random_state

        if self.criterion == 'entropy':
            self.impurity_func = self._entropy
        elif self.criterion == 'gini':
            self.impurity_func = self._gini_index
        elif self.criterion == 'gain_ratio':
            self.impurity_func = self._gain_ratio

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.tree_ = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(x, self.tree_) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        # Verificar se atingiu a condição de parada
        if n_samples < self.min_samples_split or depth == self.max_depth or len(np.unique(y)) == 1:
            return self._leaf_node(y)

        # Escolher o melhor atributo para separar os dados
        best_feature, best_threshold = self._choose_split(X, y)

        # Verificar se a árvore deve ser podada antes de continuar a construção
        if self.pre_pruning is not None and self.pre_pruning(X, y, best_feature, best_threshold):
            return self._leaf_node(y)

        # Construir subárvores para cada valor possível do atributo escolhido
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return self._decision_node(best_feature, best_threshold, left_tree, right_tree)

    def _choose_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_impurity = np.inf

        # Escolher aleatoriamente uma subconjunto de atributos, se necessário
        if self.max_features is not None:
            features_indices = np.random.choice(range(X.shape[1]), size=self.max_features, replace=False)
        else:
            features_indices = range(X.shape[1])

        for i in features_indices:
            thresholds = np.unique(X[:, i])

            for threshold in thresholds:
                left_indices = X[:, i] < threshold
                right_indices = X[:, i] >= threshold

                if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
                    continue

                impurity = (np.sum(left_indices) * self.impurity_func(y[left_indices]) +
                            np.sum(right_indices) * self.impurity_func(y[right_indices])) / len(y)

                if impurity < best_impurity:
                    best_feature = i
                    best_threshold = threshold
                    best_impurity = impurity

        return best_feature, best_threshold

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def _gini_index(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _gain_ratio(self, X, y, feature):
        feature_values, counts = np.unique(X[:, feature], return_counts=True)
        probs = counts / len(y)
        entropy = np.sum([-p * np.log2(p) for p in probs])

        split_entropy = 0
        for value in feature_values:
            indices = X[:, feature] == value
            split_entropy += np.sum(indices) / len(y) * self._entropy(y[indices])

        if entropy == 0 or split_entropy == 0:
            return 0

        information_gain = entropy - split_entropy
        intrinsic_value = -np.sum(probs * np.log2(probs))
        return information_gain / intrinsic_value

    def _leaf_node(self, y):
        counts = np.bincount(y)
        probas = counts / len(y)
        return {'leaf': True, 'probas': probas}

    def _decision_node(self, feature, threshold, left_subtree, right_subtree):
        return {'leaf': False, 'feature': feature, 'threshold': threshold,
                'left': left_subtree, 'right': right_subtree}

    def _predict(self, x, tree):
        if tree['leaf']:
            return np.argmax(tree['probas'])

        if x[tree['feature']] < tree['threshold']:
            return self._predict(x, tree['left'])
        else:
            return self._predict(x, tree['right'])


def test_decision_tree( test_size=0.2, random_state=42):
    dt = pd.read_csv('tennis.csv')
    X = dt.drop('play', axis=1)
    y = dt['play']
    # Separar os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Instanciar e treinar o modelo
    dt = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_split=5, min_samples_leaf=5)
    dt.fit(X_train, y_train)

    # Fazer previsões na base de teste
    y_pred = dt.predict(X_test)

    # Calcular a acurácia
    acc = accuracy_score(y_test, y_pred)
    print(f'Acurácia na base de teste: {acc:.2f}')

if __name__ == '__main__':
    test_decision_tree()
