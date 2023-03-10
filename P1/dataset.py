import numpy as np


class Dataset:

    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])
        self.features = []
        self.label = ''

    def get_X(self):
        return self.X

    def set_X(self, new):
        self.X = new

    def get_y(self):
        return self.y

    def get_features(self):
        return self.features

    def get_label(self):
        return self.label

    def read_csv(self, file, label=None, delimiter=','):
        data = np.genfromtxt(file, delimiter=delimiter, names=True, dtype=None, encoding=None)
        self.features = list(data.dtype.names)
        self.label = self.features[len(self.features) - 1]
        self.features.remove(self.label)
        self.X = np.vstack([data[f] for f in self.features]).T
        self.y = data[self.label]

    def read_tsv(self, file, label=None):
        self.read_csv(file, label, '\t')

    def write_csv(self, file, delimiter=','):
        data = np.hstack((self.X, self.y.reshape(-1, 1)))
        header = self.features + [self.label]
        fmt = ["%.18e" if col.dtype.kind in {'f', 'c'} else "%s" for col in data.T]
        np.savetxt(file, data, delimiter=delimiter, header=delimiter.join(header), fmt=fmt, comments='')

    def write_tsv(self, file):
        self.write_csv(file, '\t')

    def describe(self):
        # Describes the input and output variables
        for i, feature in enumerate(self.features):
            print(feature)
            collumnX = self.X[:, i]
            if np.issubdtype(collumnX.dtype, np.number):
                printNumeric(collumnX)
            else:
                printDescrite(collumnX)

        if self.label is not None:
            print(self.label)
            collumnY = self.y
            if np.issubdtype(collumnY.dtype, np.number):
                printNumeric(collumnY)
            else:
                printDescrite(collumnY)

    def replace_to_null(self, value):
        # Replaces the chosen number with null
        self.X = np.where(self.X == value, np.nan, self.X)
        self.y = np.where(self.y == value, np.nan, self.y)

    def replace_nulls(self, value):
        # Replaces every null value with the one chosen
        self.X = np.where(self.X != self.X, value, self.X)
        self.y = np.where(self.y != self.y, value, self.y)

    def count_nulls(self):
        # Counts the null values in each column
        null_count = np.zeros(self.X.shape[1], dtype=int)

        for i in range(self.X.shape[1]):
            for val in self.X[:, i]:
                if val == '' or val is None:
                    null_count[i] += 1
                elif np.issubdtype(type(val), np.number) and np.isnan(val):
                    null_count[i] += 1

        for feature in range(len(self.features)):
            print(self.features[feature], "- Null Values:", null_count[feature])
            if null_count[feature] == len(self.X):
                print(" - All values are null.")
                print(self.label, "- Null Values:", null_count[-1])
                if null_count[-1] == len(self.y):
                    print(" - All values are null.")

    def replace_nulls_with_mean(self):
        # Replaces nulls with the mean for numeric values and with the most frequent value for categorical values
        for i, feature in enumerate(self.features):
            var = self.X[:, i]
            if np.issubdtype(var.dtype, np.number):
                val = np.nanmean(var)
            else:
                unique_vals, counts = np.unique(var[var == var], return_counts=True)
                val = unique_vals[np.argmax(counts)]
            self.X[:, i] = np.where(np.logical_or(np.isnan(var), np.isinf(var)), val, var)

        var = self.y
        if np.issubdtype(var.dtype, np.number):
            val = np.nanmean(var)
            self.y = np.where(np.logical_or(np.isnan(var), np.isinf(var)), val, var)
        else:
            unique_vals, counts = np.unique(var[var == var], return_counts=True)
            val = unique_vals[np.argmax(counts)]
            self.y = np.where(var != var, val, var)


def printNumeric(collumn):
    print(" -Mean: ", np.nanmean(collumn))
    print(" -Median: ", np.nanmedian(collumn))
    print(" -Standard Deviation: ", format(np.std(collumn), '.4f'))
    print(" -Minimum: ", np.nanmin(collumn))
    print(" -Maximum: ", np.nanmax(collumn))


def printDescrite(collumn):
    unique_vals, counts = np.unique(collumn[collumn == collumn], return_counts=True)
    print(" -Number of unique values: ", len(unique_vals))
    print(" -Most frequent value: ", unique_vals[np.argmax(counts)])
