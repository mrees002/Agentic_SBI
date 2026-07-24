import numpy as np

class UniformPrior:
    def __init__(self, bounds):
        # store the bounds, which should be a dictionary with parameter names as keys and (lower, upper) tuples as values
        # i.e. {intercept: (-5, 5), ...}
        for name, interval in bounds.items():
            if len(interval) != 2:
                raise ValueError(
                    f"Bounds for '{name}' must contain two values."
                )

            lower, upper = interval

            if lower >= upper:
                raise ValueError(
                    f"Lower bound for '{name}' must be less than upper bound."
                )

        self.bounds = bounds
    
    def sample(self, rng):

        """
        Samples from the uniform prior distribution defined by the bounds.

        Parameters:
        name: string of the name of the parameter to sample
        lower: float of the lower bound of the uniform distribution
        upper: float of the upper bound of the uniform distribution
        rng: numPy generator

        Returns:
        dict: parameter names and their sampled values
        """

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator.")

        parameters = {}

        # iterate through dictionary of bounds and sample from uniform distribution for each parameter
        for name, (lower, upper) in self.bounds.items():
            parameters[name] = rng.uniform(lower, upper)
            
        return parameters
    
class GaussianPrior:

    def __init__(self, means, stds):
        # store the means and standard deviations which should be dictionaries with parameter names as keys and values corresponding
        # i.e. means = {intercept: 1, slope: 2}
        self.means = means
        self.stds = stds

        if self.means.keys() != self.stds.keys():
            raise ValueError(
                "Mean and standard deviation parameter names must match."
            )

        for name, value in self.stds.items():
            if value <= 0:
                raise ValueError(
                    f"Standard deviation for '{name}' must be positive."
                )


    def sample(self, rng):

        """
        Samples from the gaussian prior distribution defined by the the mean and standard deviations

        Parameters:
        mean: a dictionary of means of the parameters
        std: a dictionary of standard deviations of the parameters
        rng: numPy generator

        Returns:
        dict: parameter names and their sampled values
        """

        parameters = {}

        # iterate through dictionary of means/stds and sample from Gaussian distribution for each parameter
        for name, mean in self.means.items():
            parameters[name] = rng.normal(mean, self.stds[name])

        return parameters