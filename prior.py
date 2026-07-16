import numpy as np

class UniformPrior:
    def __init__(self, bounds):
        # store the bounds, which should be a dictionary with parameter names as keys and (lower, upper) tuples as values
        self.bounds = bounds
    
    def sample(self):

        """
        Samples from the uniform prior distribution defined by the bounds.

        Parameters:
        name: string of the name of the parameter to sample
        lower: float of the lower bound of the uniform distribution
        upper: float of the upper bound of the uniform distribution

        Returns:
        dict: parameter names and their sampled values
        """

        parameters = {}

        # iterate through dictionary of bounds and sample from uniform distribution for each parameter
        for name, (lower, upper) in self.bounds.items():
            parameters[name] = np.random.uniform(lower, upper)
            
        return parameters
    
class GaussianPrior:

    """
    Samples from the gaussian prior distribution defined by the the mean and standard deviations

    Parameters:
    mean: a dictionary of means of the parameters
    std: a dictionary of standard deviations of the parameters

    Returns:
    dict: parameter names and their sampled values
    """

    def __init__(self, mean, std):
        self.means = mean
        self.stds = std

    def sample(self):
        parameters = {}

        # iterate through dictionary of means/stds and sample from Gaussian distribution for each parameter
        for name, mean in self.means.items():
            parameters[name] = np.random.normal(mean, self.stds[name])

        return parameters