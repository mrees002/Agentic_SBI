import numpy as np

def mean_std_summary(y):
    """
    Computes the mean and standard deviation of a given array.

    Parameters:
    y: array of data

    Returns:
    array: mean and standard deviation of the data
    """
    
    # create arrays
    y = np.asarray(y, dtype=float)

    # run error checks
    if y.ndim != 1:
        raise ValueError("Input data must be one-dimensional.")

    if y.size < 2:
        raise ValueError("At least two observations are required.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Input data contains non-finite values.")

    # return mean and std
    return np.array([
        np.mean(y),
        np.std(y, ddof=1),
    ])

def regression_summary(x, y):

    """
    Computes the regression coefficients of a linear regression model given input data x and output data y

    Parameters:
    x: array of input data
    y: array of output data

    Returns:
    array: regression coefficients
    """

    # create arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # run error checks
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must both be one-dimensional arrays.")

    if x.size != y.size:
        raise ValueError("x and y must have the same number of observations.")

    if x.size < 2:
        raise ValueError("At least two observations are required.")

    # Add a column of ones to x for the intercept term to create design matrix
    design_matrix = np.column_stack((np.ones(x.shape[0]), x))

    # Perform linear regression using least squares
    intercept, slope = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
    
    return np.array([intercept, slope])

def regression_summary_with_sd(x, y):

    """
    Computes the regression coefficients of a linear regression model given input data x and output data y

    Parameters:
    x: array of input data
    y: array of output data

    Returns:
    array: regression coefficients
    """

    # create arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # run error checks
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must both be one-dimensional arrays.")

    if x.size != y.size:
        raise ValueError("x and y must have the same number of observations.")

    if x.size < 3:
        raise ValueError("At least three observations are required.")

    # Add a column of ones to x for the intercept term to create design matrix
    design_matrix = np.column_stack((np.ones(x.shape[0]), x))

    # Perform linear regression using least squares
    intercept, slope = np.linalg.lstsq(design_matrix, y, rcond=None)[0]

    # calculate regression line
    fitted_values = slope * x + intercept

    # get residuals
    residuals = y - fitted_values

    # calculate residual standard deviation
    residual_sd = np.sqrt(np.sum(residuals ** 2) / (x.size - 2))
    
    return np.array([intercept, slope, residual_sd])