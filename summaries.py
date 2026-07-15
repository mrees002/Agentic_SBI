import numpy as np

def mean_std_summary(y):
    """
    Computes the mean and standard deviation of a given array.

    Parameters:
    y: array of data

    Returns:
    array: mean and standard deviation of the data
    """
    mean = np.mean(y)
    std = np.std(y)
    return np.array([mean, std])

def regression_summary(x, y):

    """
    Computes the regression coefficients of a linear regression model given input data x and output data y

    Parameters:
    x: array of input data
    y: array of output data

    Returns:
    array: regression coefficients
    """

    # Add a column of ones to x for the intercept term
    X = np.column_stack((np.ones(x.shape[0]), x))

    # Perform linear regression using least squares
    intercept, slope = np.linalg.lstsq(X, y, rcond=None)[0]

    # Calculate the residuals and their standard deviation
    fitted = intercept + slope * x
    residuals = y - fitted
    residual_sd = np.sqrt(np.sum(residuals**2) / (x.size - 2))
    
    return np.array([intercept, slope, residual_sd])