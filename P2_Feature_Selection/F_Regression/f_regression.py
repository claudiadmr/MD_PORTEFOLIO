import numpy as np
import scipy.stats


def f_regression(dataset):

    # Split the data into features (X) and target (y)
    X = dataset.X
    y = dataset.y

    # Add a column of ones to X for the intercept term
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    # Compute the coefficients using linear regression
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Compute the predicted values and residuals
    y_pred = np.dot(X, beta)
    residuals = y - y_pred

    # Compute the total sum of squares
    SST = np.sum((y - np.mean(y)) ** 2)

    # Compute the residual sum of squares
    SSR = np.sum(residuals ** 2)

 
    # Compute the degrees of freedom
    n = X.shape[0]
    p = dataset.X.shape[1]  # number of features in the dataset
    df_reg = p
    df_res = n - p - 1

    # Compute the F-value and p-value
    F = (SST - SSR) / df_reg / (SSR / df_res)
    p_value = 1 - scipy.stats.f.cdf(F, df_reg, df_res)

    return F, p_value

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


def test():
    # Create a sample dataset
    np.random.seed(42)
    X = np.random.randn(1000, 3)
    y = X[:, 0] + 2 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(1000)

    dataset = Dataset(X, y)

    # Calculate F-value and p-value
    F, p_value = f_regression(dataset)

    # Print the results
    print("F-value:", F)
    print("p-value:", p_value)


if __name__ == "__main__":
    # Run the test
    test()





'''
This function takes a dataset object and performs a linear regression analysis on it to obtain an F-statistic
and a p-value for each feature. The F-statistic is used to test whether all the coefficients in the regression
model are zero, while the p-value represents the probability of obtaining the observed F-statistic (or a more 
extreme one) under the null hypothesis.

The function starts by splitting the dataset into features (X) and target (y). It then adds a column of ones to
X for the intercept term, and computes the coefficients of the linear regression model using least squares. The
predicted values and residuals are also computed.

The function then computes the total sum of squares (SST) and the residual sum of squares (SSR), which are used 
to calculate the F-statistic. The degrees of freedom for the regression and residual terms are also computed, 
as well as the number of features in the dataset.

Finally, the function computes the F-value and the p-value using the computed statistics and the degrees of freedom.
The p-value is calculated using the cumulative distribution function of an F-distribution with df_reg and df_res degrees
of freedom. The function returns the F-value and the p-value for each feature in the dataset.
'''

