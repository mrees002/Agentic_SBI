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
    """

    # Compute summary statistics for the observed data
    observed_summary = summary_fn(observed_data)
    accepted_parameters = []
    accepted_distances = []
    
    # Run simulations
    for _ in range(n_simulations):
        # Sample parameters from the prior distribution
        parameters = prior.sample()

        # Simulate data using the sampled parameters
        simulated_data = simulator(parameters)

        # Compute summary statistics for simulated data
        simulated_summary = summary_fn(simulated_data)

        # Compute the distance between the summary statistics
        distance = distance_fn(observed_summary, simulated_summary)

        # Accept parameters if the distance is below a certain threshold
        if distance <= epsilon:  # This threshold can be adjusted based on the problem
            accepted_parameters.append(parameters)
            accepted_distances.append(distance)

    return accepted_parameters, np.array(accepted_distances)