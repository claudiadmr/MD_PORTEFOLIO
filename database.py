import numpy as np
import os


class Dataset:
    def __init__(self,filename, X=None, y=None, feature_names=None, label_name=None):
        if filename is not None:
            self.read_csv(filename)

        self.X = np.array(X, dtype=[(name, np.float64) for name in feature_names]) \
            if feature_names \
            else np.array(X, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.feature_names = feature_names
        self.label_name = label_name

    def get_X(self):
        return self.X

    def set_X(self, X):
        self.X = X

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    def read_csv(self, file_path, delimiter=","):
        data = np.genfromtxt(file_path, delimiter=delimiter, names=True, dtype=None)
        X = data[list(data.dtype.names[:-1])]
        y = data[data.dtype.names[-1]]
        feature_names = X.dtype.names
        label_name = y.dtype.names[0]
        return Dataset(X, y, feature_names, label_name)

    def write_csv(self, file_path, delimiter=","):
        header = ",".join(self.feature_names) + "," + self.label_name + "\n"
        np.savetxt(file_path, np.hstack((self.X, self.y.reshape(-1, 1))), delimiter=delimiter, header=header,
                   comments="")

    def describe(self):
        means = np.mean(self.X, axis=0)
        stds = np.std(self.X, axis=0)
        mins = np.min(self.X, axis=0)
        maxs = np.max(self.X, axis=0)
        for i, name in enumerate(self.feature_names):
            print("Feature:", name)
            print("  Mean:", means[i])
            print("  Std:", stds[i])
            print("  Min:", mins[i])
            print("  Max:", maxs[i])

    def count_null(self):
        null_counts = np.sum(np.isnan(self.X), axis=0)
        for i, name in enumerate(self.feature_names):
            print("Feature:", name)
            print("  Null count:", null_counts[i])

    def replace_null(self, method="mean"):
        if method == "mean":
            X_filled = np.nan_to_num(self.X, nan=np.nanmean(self.X, axis=0))
        elif method == "mode":
            X_filled = self.X.copy()
            for i, name in enumerate(self.feature_names):
                mode = np.nanargmax(np.bincount(X_filled[:, i].astype(int)))
                X_filled[:, i][np.isnan
                (self.X[:, i])] = mode
        else:
            raise ValueError("Invalid method. Please choose 'mean' or 'mode'.")
        return Dataset(X_filled, self.y, self.feature_names, self.label_name)


def teste():
    dataset = Dataset('/Users/cdmr/Desktop/univercidade/1ano/2semestre/MD/1aula/notas.csv')
    print(os.getcwd())  # Imprime o diretório
    dataset.describe()


# main - tests
if __name__ == '__main__':
    teste()
