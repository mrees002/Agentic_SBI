import numpy as np

def simulate_linear_regression(theta, x, noise_sd, rng):
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

    x = np.asarray(x, dtype=float)

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
    
    intercept = float(theta["intercept"])
    slope = float(theta["slope"])

    if not np.isfinite(intercept) or not np.isfinite(slope):
        raise ValueError("theta values must be finite.")

    mean = intercept + slope * x
    return rng.normal(mean, noise_sd)

def simulate_exponential_decay(theta, time, noise_sd, rng):
    """
    A simple exponential decay simulator that generates data based on the provided parameters.

    Parameters: 
    theta: dictionary of parameters with keys 'initial_amplitude' and 'decay_rate'
    x: array of input data
    noise_sd: standard deviation of the Gaussian noise to be added to the output
    rng: numPy generator

    Returns:
    array: simulated output based on exponential decay model with gaussian noise
    """
    
    time = np.asarray(time, dtype=float)

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

    initial_value = float(theta["initial_value"])
    decay_rate = float(theta["decay_rate"])

    if not np.isfinite(initial_value) or not np.isfinite(decay_rate):
        raise ValueError("theta values must be finite.")

    mean = initial_value * np.exp(-decay_rate * time)
    return rng.normal(mean, noise_sd)