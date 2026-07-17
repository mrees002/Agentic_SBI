import numpy as np

def exact_linear_regression_posterior(x, y, noise_sd, prior_mean, prior_covariance):

    """
    Computes the exact posterior distribution for a linear regression model with Gaussian noise.

    Parameters:
    x: array of x values
    y: array of observed y values
    noise_sd: standard deviation of the Gaussian noise
    prior: a GaussianPrior instance, providing means and stds for intercept and slope

    Returns:
    mu_n: posterior mean of the parameters (intercept and slope)
    sigma_n: posterior covariance matrix of the parameters
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_covariance = np.asarray(prior_covariance, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    if noise_sd <= 0 or not np.isfinite(noise_sd):
        raise ValueError("noise_sd must be a positive finite number.")

    if prior_mean.shape != (2,):
        raise ValueError("prior_mean must have shape (2,).")

    if prior_covariance.shape != (2, 2):
        raise ValueError("prior_covariance must have shape (2, 2).")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values.")

    if not np.all(np.isfinite(prior_mean)):
        raise ValueError("prior_mean must contain only finite values.")

    if not np.all(np.isfinite(prior_covariance)):
        raise ValueError(
            "prior_covariance must contain only finite values."
        )

    design_matrix = np.column_stack(
        (np.ones(x.size), x)
    )

    prior_precision = np.linalg.inv(prior_covariance)
    noise_variance = noise_sd ** 2

    posterior_precision = (
        prior_precision
        + design_matrix.T @ design_matrix / noise_variance
    )

    posterior_covariance = np.linalg.inv(
        posterior_precision
    )

    posterior_mean = posterior_covariance @ (
        prior_precision @ prior_mean
        + design_matrix.T @ y / noise_variance
    )

    return posterior_mean, posterior_covariance

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