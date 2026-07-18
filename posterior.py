import numpy as np

def exact_linear_regression_posterior(x, y, noise_sd, prior_mean, prior_covariance):

    """
    Computes the exact posterior distribution for a linear regression model with Gaussian noise.

    Parameters:
    x: array of x values
    y: array of observed y values
    noise_sd: standard deviation of the Gaussian noise
    prior_mean: prior mean for intercept and slope
    prior_covariance: 2 x 2 prior covariance matrix

    Returns:
    posterior_mean: posterior mean for intercept and slope
    posterior_covariance: posterior covariance matrix
    """

    # ensure everything is an array
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_covariance = np.asarray(prior_covariance, dtype=float)

    # run error checks
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
    
    # ensure covaraince matrix is invertible
    if not np.allclose(prior_covariance, prior_covariance.T):
        raise ValueError("prior_covariance must be symmetric.")

    try:
        np.linalg.cholesky(prior_covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "prior_covariance must be positive definite and invertible."
        ) from error

    # create design matrix
    design_matrix = np.column_stack(
        (np.ones(x.size), x)
    )

    # find prior precision and create variance
    prior_precision = np.linalg.inv(prior_covariance)
    noise_variance = noise_sd ** 2

    # find posterior precision
    posterior_precision = (
        prior_precision
        + design_matrix.T @ design_matrix / noise_variance
    )

    # ensure posterior precision matrix is invertible
    try:
        posterior_covariance = np.linalg.inv(
            posterior_precision
        )
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Posterior precision matrix is not invertible."
        ) from error

    # find posterior covariance
    posterior_covariance = np.linalg.inv(
        posterior_precision
    )

    # calculate posterior mean
    posterior_mean = posterior_covariance @ (
        prior_precision @ prior_mean
        + design_matrix.T @ y / noise_variance
    )

    return posterior_mean, posterior_covariance

def sample_exact_posterior(posterior_mean, posterior_covariance, n_samples, rng):

    """
    Samples from the exact posterior distribution for a linear regression model with Gaussian noise.

    Parameters:
    mu_n: posterior mean of the parameters (intercept and slope)
    sigma_n: posterior covariance matrix of the parameters
    n_samples: number of samples to draw
    rng: numPy generator

    Returns:
    A dictionary with the sampled intercept and slope values
    """

    # run error checks

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    # take a multivariate normal sample from the posterior distribution of the parameters
    samples = rng.multivariate_normal(posterior_mean, posterior_covariance, n_samples)
    
    # return a dictionary with the sampled intercept and slope values
    return {
        "intercept": samples[:, 0],
        "slope": samples[:, 1]
    }