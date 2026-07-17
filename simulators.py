import numpy as np

def simulate_linear_regression_sigma(theta, x, noise_sd, rng):
    """
    A simple linear regression simulator that generates data based on the provided parameters.

    Parameters: 
    theta: dictionary of parameters with keys 'intercept' and 'slope'
    x: array of input data
    noise_sd: standard deviation of the Gaussian noise to be added to the output
    rng: numPy generator

    Returns:
    array: simulated output based on linear model with gaussian noise
    """

    # create array
    x = np.asarray(x, dtype=float)

    # run error checks
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")

    if noise_sd <= 0:
        raise ValueError("noise_sd must be positive.")

    required = {"intercept", "slope"}
    if set(theta) != required:
        raise ValueError(
            "theta must contain exactly 'intercept' and 'slope'."
        )
    
    # get intercept and slope
    intercept = float(theta["intercept"])
    slope = float(theta["slope"])

    if not np.isfinite(intercept) or not np.isfinite(slope):
        raise ValueError("theta values must be finite.")
    
    # get mean
    mean = intercept + slope * x
    return rng.normal(mean, noise_sd)

def simulate_linear_regression_unknown_sigma(theta, x, rng):
    """
    A simple linear regression simulator that generates data based on the provided parameters.

    Parameters: 
    theta: dictionary of parameters with keys 'intercept', 'slope', and 'standard_deviation'
    x: array of input data
    rng: numPy generator

    Returns:
    array: simulated output based on linear model with gaussian noise
    """

    # create array
    x = np.asarray(x, dtype=float)

    # run error checks
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")

    required = {"intercept", "slope", "standard_deviation"}
    if set(theta) != required:
        raise ValueError(
            "theta must contain exactly 'intercept' and 'slope'."
        )
    
    # get intercept and slope
    intercept = float(theta["intercept"])
    slope = float(theta["slope"])
    standard_deviation = float(theta["standard_deviation"])

    if not np.isfinite(intercept) or not np.isfinite(slope):
        raise ValueError("theta values must be finite.")
    
    # get mean
    mean = intercept + slope * x
    return rng.normal(mean, standard_deviation)

def simulate_exponential_decay(theta, time, noise_sd, rng):
    """
    A simple exponential decay simulator that generates data based on the provided parameters.

    Parameters: 
    theta: dictionary of parameters with keys 'initial_amplitude' and 'decay_rate'
    time: array of input data
    noise_sd: standard deviation of the Gaussian noise to be added to the output
    rng: numPy generator

    Returns:
    array: simulated output based on exponential decay model with gaussian noise
    """
    
    # create array for time
    time = np.asarray(time, dtype=float)

    # run error checks
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")

    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")

    if noise_sd <= 0:
        raise ValueError("noise_sd must be positive.")

    required = {"initial_value", "decay_rate"}
    if set(theta) != required:
        raise ValueError(
            "theta must contain exactly 'initial_value' and 'decay_rate'."
        )

    # get values
    initial_value = float(theta["initial_value"])
    decay_rate = float(theta["decay_rate"])

    if not np.isfinite(initial_value) or not np.isfinite(decay_rate):
        raise ValueError("theta values must be finite.")

    # get mean
    mean = initial_value * np.exp(-decay_rate * time)
    return rng.normal(mean, noise_sd)