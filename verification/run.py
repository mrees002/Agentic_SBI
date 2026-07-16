import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from posterior import exact_posterior, sample_exact_posterior
from prior import UniformPrior
from inference import abc_function
from simulators import simulate_exponential_decay, simulate_linear_regression
from summaries import mean_std_summary, regression_summary
from distance import euclidean_distance
from plots import plot_comparison, plot_observed_data, plot_posterior

import numpy as np

# Settings
N_OBSERVATIONS = 50
TRUE_INTERCEPT = 1.0
TRUE_SLOPE = 2.0

TRUE_AMPLITUDE = 3.0
TRUE_DECAY_RATE = 0.5

NOISE_SD = 0.5
EPSILON = 0.25
N_SIMULATIONS = 100000

def linear_regression_experiment():

    # Getting observed data using a linear regression simulator with known parameters
    x = np.linspace(0, 1, N_OBSERVATIONS)
    true_theta = {"intercept": TRUE_INTERCEPT, "slope": TRUE_SLOPE}
    observed_data = simulate_linear_regression(true_theta, x, NOISE_SD)

    # Define the simulator function for ABC using the linear regression simulator
    def simulator_for_abc(theta):
        return simulate_linear_regression(theta, x, NOISE_SD)

    # Define the summary function for ABC using the regression summary
    def summary_for_abc(y):
        return regression_summary(x, y)

    # Run ABC inference
    accepted_parameters, accepted_distances = abc_function(UniformPrior(bounds={'intercept': (-10, 10), 'slope': (-10, 10)}), 
                                                        simulator_for_abc, observed_data, summary_for_abc, euclidean_distance, 
                                                        EPSILON, N_SIMULATIONS)

    # Print results
    print(f"Accepted {len(accepted_parameters)} samples out of {N_SIMULATIONS}")
    print(f"Mean intercept: {np.mean([p['intercept'] for p in accepted_parameters]):.3f}")
    print(f"Mean slope: {np.mean([p['slope'] for p in accepted_parameters]):.3f}")

    # Get the exact posterior estimates
    mu_n, sigma_n = exact_posterior(x, observed_data, NOISE_SD)
    exact_samples = sample_exact_posterior(mu_n, sigma_n, n_samples=len(accepted_parameters))

    # Plotting the results
    plot_observed_data(x, observed_data, TRUE_INTERCEPT, TRUE_SLOPE)
    plot_posterior(accepted_parameters, true_theta)
    plot_comparison(accepted_parameters, exact_samples, TRUE_INTERCEPT, TRUE_SLOPE)

def exponential_decay_experiment():

    # Getting observed data using an exponential decay simulator with known parameters
    x = np.linspace(0, 5, N_OBSERVATIONS)
    true_theta = {"initial_amplitude": TRUE_AMPLITUDE, "decay_rate": TRUE_DECAY_RATE}
    observed_data = simulate_exponential_decay(true_theta, x, NOISE_SD)

    # Define the simulator function for ABC using the exponential decay simulator
    def simulator_for_abc(theta):
        return simulate_exponential_decay(theta, x, NOISE_SD)

    # Define the summary function for ABC using the regression summary
    def summary_for_abc(y):
        return mean_std_summary(y)

    # Run ABC inference
    accepted_parameters, accepted_distances = abc_function(UniformPrior(bounds={'initial_amplitude': (0, 10), 'decay_rate': (0, 10)}), 
                                                        simulator_for_abc, observed_data, summary_for_abc, euclidean_distance, 
                                                        EPSILON, N_SIMULATIONS)

    # Print results
    print(f"Accepted {len(accepted_parameters)} samples out of {N_SIMULATIONS}")
    print(f"Mean initial amplitude: {np.mean([p['initial_amplitude'] for p in accepted_parameters]):.3f}")
    print(f"Mean decay rate: {np.mean([p['decay_rate'] for p in accepted_parameters]):.3f}")

    # Plotting the results
    plot_posterior(accepted_parameters, {"initial_amplitude": TRUE_AMPLITUDE, "decay_rate": TRUE_DECAY_RATE})

if __name__ == "__main__":
    linear_regression_experiment()