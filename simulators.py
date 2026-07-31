import numpy as np

def simulate_linear_regression(slope, intercept, x, sd, rng):
    """
    A simple linear regression simulator that generates data based on the provided parameters.

    Parameters: 
    slope: slope of the regression simulator
    intercept: intercept of the regression simulator
    x: array of input data
    sd: standard deviation of the Gaussian noise to be added to the output
    rng: numPy generator

    Returns:
    array: simulated output based on linear model with gaussian noise
    """

    x = np.asarray(x)

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numPy generator")

    if sd < 0:
        raise ValueError("Standard Deviation must be greater than 0")
    
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
        raise ValueError("Standard Deviation must be greater than 0")

    return rng.normal(mean, std, sample)

def simulate_exponential_decay(initial_value, decay_rate, time, sd, rng):
    """
    A simple exponential decay simulator that generates data based on the provided parameters.

    Parameters: 
    intial_value: initial value of the simulator
    decay_rate: decay rate of the simulator
    time: array of input data
    sd: standard deviation of the Gaussian noise to be added to the output
    rng: numPy generator

    Returns:
    array: simulated output based on exponential decay model with gaussian noise
    """

    time = np.asarray(time)

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numPy generator")

    if sd < 0:
        raise ValueError("Standard Deviation must be greater than 0")

    # get mean
    mean = initial_value * np.exp(-decay_rate * time)
    return rng.normal(mean, sd)