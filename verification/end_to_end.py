import os
import sys

import numpy as np

# Allow imports from the repository root
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

from distance import euclidean_distance
from inference import abc_function
from prior import UniformPrior
from simulators import simulate_linear_regression
from summaries import regression_summary


# Make the example reproducible
np.random.seed(42)


# Fixed regression inputs
x = np.linspace(0, 1, 25)
noise_sd = 0.2


# Parameters used to generate synthetic observed data
true_theta = {
    "intercept": 1.0,
    "slope": 2.0,
}


# Generate one observed dataset
observed_data = simulate_linear_regression(
    theta=true_theta,
    x=x,
    noise_sd=noise_sd,
)


# Prior used by ABC
prior = UniformPrior(
    bounds={
        "intercept": (-1.0, 3.0),
        "slope": (0.0, 4.0),
    }
)


# Adapt the general simulator to the interface expected by abc_function
def simulator_for_abc(theta):
    return simulate_linear_regression(
        theta=theta,
        x=x,
        noise_sd=noise_sd,
    )


# Adapt the regression summary to the interface expected by abc_function
def summary_for_abc(y):
    return regression_summary(x, y)


# Run rejection ABC
accepted_parameters, accepted_distances = abc_function(
    prior=prior,
    simulator=simulator_for_abc,
    observed_data=observed_data,
    summary_fn=summary_for_abc,
    distance_fn=euclidean_distance,
    epsilon=0.25,
    n_simulations=20_000,
)


print(f"Accepted samples: {len(accepted_parameters)}")
print(
    f"Acceptance rate: "
    f"{len(accepted_parameters) / 20_000:.3%}"
)


if accepted_parameters:
    intercept_samples = np.array(
        [
            sample["intercept"]
            for sample in accepted_parameters
        ]
    )

    slope_samples = np.array(
        [
            sample["slope"]
            for sample in accepted_parameters
        ]
    )

    print()
    print("True parameters")
    print(f"Intercept: {true_theta['intercept']}")
    print(f"Slope: {true_theta['slope']}")

    print()
    print("ABC posterior means")
    print(f"Intercept: {np.mean(intercept_samples):.3f}")
    print(f"Slope: {np.mean(slope_samples):.3f}")

    print()
    print("ABC 95% intervals")
    print(
        "Intercept:",
        np.quantile(intercept_samples, [0.025, 0.975]),
    )
    print(
        "Slope:",
        np.quantile(slope_samples, [0.025, 0.975]),
    )

    print()
    print(
        f"Smallest accepted distance: "
        f"{min(accepted_distances):.4f}"
    )
else:
    print(
        "No samples were accepted. "
        "Try increasing epsilon."
    )