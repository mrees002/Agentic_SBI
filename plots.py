from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_posterior(
    accepted_parameters,
    true_values=None,
    output_path=None,
):
    """
    Plot marginal posterior histograms for accepted ABC parameters.
    """
    if len(accepted_parameters) == 0:
        raise ValueError("No accepted samples to plot.")

    parameter_names = list(accepted_parameters[0].keys())

    for sample in accepted_parameters:
        if set(sample.keys()) != set(parameter_names):
            raise ValueError(
                "All accepted samples must contain the same parameters."
            )

    figure, axes = plt.subplots(
        nrows=len(parameter_names),
        ncols=1,
        figsize=(7, 3 * len(parameter_names)),
    )

    if len(parameter_names) == 1:
        axes = [axes]

    for axis, name in zip(axes, parameter_names):
        values = np.asarray(
            [sample[name] for sample in accepted_parameters],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Accepted values for '{name}' must be finite."
            )

        axis.hist(values, bins=30, density=True)
        axis.set_title(f"Posterior for {name}")
        axis.set_xlabel(name)
        axis.set_ylabel("Density")

        if true_values is not None and name in true_values:
            axis.axvline(
                true_values[name],
                linestyle="--",
                label="True value",
            )
            axis.legend()

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)

    return figure

def plot_observed_data(x, observed_data, true_intercept, true_slope):

    # Plotting the observed data and true regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(x, observed_data, color='blue', alpha=0.5, label='Observed Data')
    plt.plot(x, true_intercept + true_slope * x, color='red', linewidth=2, label='True Regression Line')
    plt.title('Observed Data with True Regression Line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("results/observed_data.png")
    plt.show()

def plot_comparison(abc_parameters, exact_parameters, true_intercept, true_slope):

    # Plotting the comparison of ABC and exact posterior distributions for intercept and slope
    abc_intercepts = [p["intercept"] for p in abc_parameters]
    abc_slopes = [p["slope"] for p in abc_parameters]
    exact_intercepts = exact_parameters["intercept"]
    exact_slopes = exact_parameters["slope"]

    # Plotting the comparison of posterior distributions for intercept
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(abc_intercepts, bins=30, density=True, alpha=0.5, color='blue', label='ABC Posterior')
    plt.hist(exact_intercepts, bins=30, density=True, alpha=0.5, color='orange', label='Exact Posterior')
    plt.axvline(true_intercept, color='red', linestyle='dashed', linewidth=2, label='True Intercept')
    plt.title('Comparison of Posterior Distributions for Intercept')
    plt.xlabel('Intercept')
    plt.ylabel('Density')
    plt.legend()

    # Plotting the comparison of posterior distributions for the slope
    plt.subplot(1, 2, 2)
    plt.hist(abc_slopes, bins=30, density=True, alpha=0.5, color='blue', label='ABC Posterior')
    plt.hist(exact_slopes, bins=30, density=True, alpha=0.5, color='orange', label='Exact Posterior')
    plt.axvline(true_slope, color='red', linestyle='dashed', linewidth=2, label='True Slope')
    plt.title('Comparison of Posterior Distributions for Slope')
    plt.xlabel('Slope')
    plt.ylabel('Density')
    plt.legend()

    # Plotting together
    plt.tight_layout()
    plt.savefig("results/comparison.png")
    plt.show()