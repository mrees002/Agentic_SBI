import numpy as np

def simulate_exponential_decay(theta, x, noise_sd):
    """
    A simple exponential decay simulator that generates data based on the provided parameters.

    Parameters: 
    theta: dictionary of parameters with keys 'initial_amplitude' and 'decay_rate'
    x: array of input data
    noise_sd: standard deviation of the Gaussian noise to be added to the output

    Returns:
    array: simulated output based on exponential decay model with gaussian noise
    """
    # compute the mean (initial_amplitude * exp(-decay_rate * x))
    y_mean = theta['initial_amplitude'] * np.exp(-theta['decay_rate'] * x)

    # add gaussian noise
    y_simulated = y_mean + np.random.normal(0, noise_sd, size=x.shape)

    # return the simulated y values
    return y_simulated