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