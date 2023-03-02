import numpy as np
import pandas as pd

class Dataset:

    def init(self):
        self.X = np.array([])
        self.y = np.array([])
        self.features = []
        self.label = ''

    def read_csv(self, file, label=None, delimiter=','):
        data = np.genfromtxt(file, delimiter=delimiter, names=True, dtype=None, encoding=None)
        self.features = list(data.dtype.names)
        self.label = self.features[len(self.features)-1]
        self.features.remove(self.label)
        self.X = np.vstack([data[f] for f in self.features]).T
        self.y = data[self.label]

    # Describes the input and output variables
    def describe(self):
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