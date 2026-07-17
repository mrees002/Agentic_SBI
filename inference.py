import numpy as np

def abc_function(prior, simulator, observed_data, summary_fn, distance_fn, epsilon, n_simulations):
    
    """
    This function performs Approximate Bayesian Computation 

    Parameters:
    prior: An instance of the UniformPrior class that defines the prior distribution for the parameters
    simulator: A function that takes parameters as input and returns simulated data
    observed_data: The observed data for which we want to infer the parameters
    summary_fn: Function to compute summary statistics of the data
    distance_fn: Function to compute distance between the summary statistics of observed and simulated data
    epsilon: The threshold for accepting parameters
    n_simulations: Number of simulations to run

    Returns:
    accepted_parameters: List of accepted parameter sets
    accepted_distances: np array of accepted distances
    """

    if not np.isscalar(epsilon) or not np.isfinite(epsilon):
        raise ValueError("epsilon must be a finite number.")

    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative.")

    if (
        not isinstance(n_simulations, int)
        or isinstance(n_simulations, bool)
        or n_simulations <= 0
    ):
        raise ValueError("n_simulations must be a positive integer.")

    observed_summary = np.asarray(
        summary_fn(observed_data),
        dtype=float,
    )

    if not np.all(np.isfinite(observed_summary)):
        raise ValueError(
            "Observed summary contains non-finite values."
        )

    accepted_parameters = []
    accepted_distances = []

    for _ in range(n_simulations):

        theta = prior.sample()
        simulated_data = simulator(theta)

        simulated_summary = np.asarray(
            summary_fn(simulated_data),
            dtype=float,
        )

        distance = float(
            distance_fn(observed_summary, simulated_summary)
        )

        if not np.isfinite(distance):
            raise ValueError(
                "Distance function returned a non-finite value."
            )

        if distance <= epsilon:
            accepted_parameters.append(theta)
            accepted_distances.append(distance)

    return accepted_parameters, accepted_distances