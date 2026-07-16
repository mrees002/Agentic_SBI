import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from posterior import exact_posterior, sample_exact_posterior
from prior import GaussianPrior
from inference import abc_function
from simulators import simulate_linear_regression
from summaries import regression_summary
from distance import euclidean_distance
from plots import plot_comparison, plot_observed_data, plot_posterior

import numpy as np
from scipy.stats import norm

# ABC Settings
EPSILON = 0.5
N_SIMULATIONS = 100000
N_OBSERVATIONS = 50

# Sample Settings
N_DATASETS = 5

# Regression Settings
NOISE_SD = 0.5
TRUE_INTERCEPT = 1.0
TRUE_SLOPE = 2.0

# generate fixed observed data
x = np.linspace(0, 1, N_OBSERVATIONS)
true_theta = {"intercept": TRUE_INTERCEPT, "slope": TRUE_SLOPE}
observed_data = simulate_linear_regression(true_theta, x, NOISE_SD)

# Generate gaussian prior, allows for precise calculation of exact prior
prior = GaussianPrior(mean={'intercept': 0, 'slope': 0}, std={'intercept': 5, 'slope': 5})

# Define a function to run a regression experiment with a given epsilon and simulation number
def linear_regression_experiment(epsilon, n_simulations):

    # Define the simulator function for ABC using the linear regression simulator
    def simulator_for_abc(theta):
        return simulate_linear_regression(theta, x, NOISE_SD)

    # Define the summary function for ABC using the regression summary
    def summary_for_abc(y):
        return regression_summary(x, y)

    # Run ABC inference
    accepted_parameters, accepted_distances = abc_function(prior, simulator_for_abc, observed_data, 
                                                           summary_for_abc, euclidean_distance, 
                                                           epsilon, n_simulations)

    # Get the exact posterior estimates
    mu_n, sigma_n = exact_posterior(x, observed_data, NOISE_SD, prior)
    exact_samples = sample_exact_posterior(mu_n, sigma_n, n_samples=len(accepted_parameters))

    # ABC credible interval
    abc_intercepts = [p['intercept'] for p in accepted_parameters]
    abc_ci = np.percentile(abc_intercepts, [2.5, 97.5])

    # Print results
    print(f"Accepted {len(accepted_parameters)} samples out of {n_simulations}")

    # Exact intercept mean/CI 
    exact_mean_intercept = mu_n[0]
    exact_sd_intercept = np.sqrt(sigma_n[0, 0])
    exact_ci = norm.interval(0.95, loc=exact_mean_intercept, scale=exact_sd_intercept)

    print(f"ABC intercept: {np.mean(abc_intercepts):.3f} [{abc_ci[0]:.3f}, {abc_ci[1]:.3f}]")
    print(f"Exact intercept: {exact_mean_intercept:.3f} [{exact_ci[0]:.3f}, {exact_ci[1]:.3f}]")

    exact_mean_slope = mu_n[1]
    exact_sd_slope = np.sqrt(sigma_n[1, 1])
    exact_ci_slope = norm.interval(0.95, loc=exact_mean_slope, scale=exact_sd_slope)

    abc_slopes = [p['slope'] for p in accepted_parameters]
    abc_ci_slope = np.percentile(abc_slopes, [2.5, 97.5])

    print(f"ABC slope: {np.mean(abc_slopes):.3f} [{abc_ci_slope[0]:.3f}, {abc_ci_slope[1]:.3f}]")
    print(f"Exact slope: {exact_mean_slope:.3f} [{exact_ci_slope[0]:.3f}, {exact_ci_slope[1]:.3f}]")

    # Plotting the results
    plot_observed_data(x, observed_data, TRUE_INTERCEPT, TRUE_SLOPE)
    plot_posterior(accepted_parameters, true_theta)
    plot_comparison(accepted_parameters, exact_samples, TRUE_INTERCEPT, TRUE_SLOPE)

# run regression experiment with various epsilons and simulations

# 1: small epsilon, small number of simulations
linear_regression_experiment(epsilon = 0.1, n_simulations = 10000)

# 2: large epsilon, small number of simulations
linear_regression_experiment(epsilon = 1.0, n_simulations = 10000)

# 3: small epsilon, large number of simulations
linear_regression_experiment(epsilon = 0.1, n_simulations = 100000)

# 4: large epsilon, large number of simulations
linear_regression_experiment(epsilon = 1.0, n_simulations = 100000)

# 5: medium epsilon, large number of simulations
linear_regression_experiment(epsilon = 0.5, n_simulations = 100000)