import numpy as np

class UniformPrior:
    def __init__(self, bounds):
        # store the bounds, which should be a dictionary with parameter names as keys and (lower, upper) tuples as values
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

    def __init__(self, means, stds):
        
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


    def sample(self):

        """
        Samples from the gaussian prior distribution defined by the the mean and standard deviations

        Parameters:
        mean: a dictionary of means of the parameters
        std: a dictionary of standard deviations of the parameters

        Returns:
        dict: parameter names and their sampled values
        """

        parameters = {}

        # iterate through dictionary of means/stds and sample from Gaussian distribution for each parameter
        for name, mean in self.means.items():
            parameters[name] = np.random.normal(mean, self.stds[name])

        return parameters
    

if __name__ == "__main__":
    uniform_prior = UniformPrior({
        "intercept": (-2, 2),
        "slope": (0, 5),
    })

    uniform_sample = uniform_prior.sample()
    print("Uniform sample:", uniform_sample)

    assert -2 <= uniform_sample["intercept"] <= 2
    assert 0 <= uniform_sample["slope"] <= 5

    gaussian_prior = GaussianPrior(
        {"intercept": 0, "slope": 1},
        {"intercept": 2, "slope": 0.5},
    )

    gaussian_sample = gaussian_prior.sample()
    print("Gaussian sample:", gaussian_sample)

    assert set(gaussian_sample.keys()) == {
        "intercept",
        "slope",
    }

    print("All prior checks passed.")