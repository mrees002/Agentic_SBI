import numpy as np

def simulate_linear_regression(slope, intercept, x, sd, rng):
    """
    A simple linear regression simulator that generates data based on the provided parameters.

    Parameters: 
    theta: dictionary of parameters with keys 'intercept' and 'slope'
    x: array of input data
    sd: standard deviation of the Gaussian noise to be added to the output
    rng: numPy generator

    Returns:
    array: simulated output based on linear model with gaussian noise
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numPy generator")

    if sd < 0:
        raise ValueError("Standard Deviation must be greater than or equal to 0")
    
    mean = intercept + slope * x
    return rng.normal(mean, sd)

def simulate_normal(mean, std, sample = 100, rng = None):
    """
    A simple normal distribution simulator

    Parameters:
    mean: mean of the distribution
    std: standard deviation of the distribution
    sample: sample size of the distribution
    rng: numPy generator

    Returns:
    array: simulated output of a normal distribution model
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numPy generator")

    if std < 0:
        raise ValueError("Standard Deviation must be greater than or equal to 0")

    return rng.normal(mean, std, sample)

def simulate_exponential_decay(initial_value, decay_rate, time, sd, rng):
    """
    A simple exponential decay simulator that generates data based on the provided parameters.

    Parameters: 
    theta: dictionary of parameters with keys 'initial_amplitude' and 'decay_rate'
    time: array of input data
    sd: standard deviation of the Gaussian noise to be added to the output
    rng: numPy generator

    Returns:
    array: simulated output based on exponential decay model with gaussian noise
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numPy generator")

    # get mean
    mean = initial_value * np.exp(-decay_rate * time)
    return rng.normal(mean, sd)