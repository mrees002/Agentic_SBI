import numpy as np

def exact_posterior(x, y, noise_sd):

    """
    Computers the exact posterior distribution for a linear regression model with Gaussian noise.

    Parameters:
    x: array of x values
    y: array of observed y values
    noise_sd: standard deviation of the Gaussian noise

    Returns:
    mu_n: posterior mean of the parameters (intercept and slope)
    sigma_n: posterior covariance matrix of the parameters
    """

    # Add a column of ones to x for the intercept term to create the design matrix
    X = np.column_stack((np.ones(x.shape[0]), x))

    # Define prior mean and covariance for the parameters (intercept and slope)
    mu_0 = np.array([0, 0])
    sigma_0 = np.diag([25, 25]) # We assume a prior variance of 25 for both parameters, which is equivalent to a standard deviation of 5

    # Calculate the posterior covariance matrix
    sigma_n = np.linalg.inv(np.linalg.inv(sigma_0) + X.T @ X / noise_sd ** 2)
    mu_n = sigma_n @ (np.linalg.inv(sigma_0) @ mu_0 + X.T @ y / noise_sd ** 2)

    return mu_n, sigma_n

def sample_exact_posterior(mu_n, sigma_n, n_samples):

    """
    Samples from the exact posterior distribution for a linear regression model with Gaussian noise.

    Parameters:
    mu_n: posterior mean of the parameters (intercept and slope)
    sigma_n: posterior covariance matrix of the parameters
    n_samples: number of samples to draw

    Returns:
    A dictionary with the sampled intercept and slope values
    """

    # take a multivariate normal sample from the posterior distribution of the parameters
    samples = np.random.multivariate_normal(mu_n, sigma_n, n_samples)
    
    # return a dictionary with the sampled intercept and slope values
    return {
        "intercept": samples[:, 0],
        "slope": samples[:, 1]
    }