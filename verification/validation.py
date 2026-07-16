import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prior import UniformPrior
from inference import abc_function
from simulators import simulate_linear_regression
from summaries import regression_summary
from distance import euclidean_distance

import numpy as np

# ABC Settings
EPSILON = 0.5
N_SIMULATIONS = 100000

# Sample Settings
N_DATASETS = 5

# Regression Settings
NOISE_SD = 0.5

# Set X to have 50 observations between 0 and 1
x = np.linspace(0, 1, 50)

# Set prior bounds
prior = UniformPrior(bounds={'intercept': (-10, 10), 'slope': (-10, 10)})

# Set Simulator for ABC simulation
def simulator_for_abc(theta):
    return simulate_linear_regression(theta, x, NOISE_SD)

# Set Summary statistic for ABC simulation
def summary_for_abc(y):
    return regression_summary(x, y)

# Define function to collect data with sampled true thetas
def sample_data(n_datasets, prior, simulator, x, noise_sd):
    """
    Generates multiple synthetic regression datasets with known parameters
    """
    datasets = []

    for _ in range(n_datasets):
        cur_theta = prior.sample()
        y_sim = simulator(cur_theta, x, noise_sd)
        datasets.append({'theta': cur_theta, 'y': y_sim})

    return datasets

# Define function to run ABC simulations on linear regression data
def run_sample_simulations(datasets):
    """
    Runs ABC simulation on generated synthetic datasets and prints results
    """
    # Iterate through all thetas and their data
    for i, data in enumerate(datasets):
        theta = data['theta']
        obs_data = data['y']

        true_slope, true_intercept = theta['slope'], theta['intercept']

        # Run ABC
        accepted_parameters, accepted_distances = abc_function(
            prior, simulator_for_abc, obs_data, summary_for_abc,
            euclidean_distance, EPSILON, N_SIMULATIONS
        )

        # Print Results
        print(f"Dataset {i+1}")
        print(f"Accepted {len(accepted_parameters)} samples out of {N_SIMULATIONS}")
        print(f"Mean intercept: {np.mean([p['intercept'] for p in accepted_parameters]):.3f}")
        print(f"Mean slope: {np.mean([p['slope'] for p in accepted_parameters]):.3f}")
        print(f"True intercept was {true_intercept}")
        print(f"True slope was {true_slope}")
        print()

# Get datasets with true thetas
datasets = sample_data(N_DATASETS, prior, simulate_linear_regression, x, NOISE_SD)

# Run ABC simulations
run_sample_simulations(datasets)