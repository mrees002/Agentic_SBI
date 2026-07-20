import matplotlib.pyplot as plt
import numpy as np


def plot_posterior(accepted_parameters, true_values=None, output_path=None):
    """Plot marginal posterior histograms for accepted ABC parameters."""
    names = list(accepted_parameters[0].keys())

    plt.figure(figsize=(7, 3 * len(names)))

    for i, name in enumerate(names, start=1):
        values = [sample[name] for sample in accepted_parameters]

        plt.subplot(len(names), 1, i)
        plt.hist(values, bins=30, density=True, alpha=0.7, label="Approximate Posterior")

        if true_values is not None and name in true_values:
            plt.axvline(true_values[name], color="red", linestyle="--", label="True value")

        plt.title(f"Posterior for {name}")
        plt.xlabel(name)
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)

    plt.show()


def plot_observed_data(x, observed_data, true_intercept, true_slope, output_path=None):
    """Plot observed regression data and the true regression line."""
    true_line = true_intercept + true_slope * np.array(x)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, observed_data, alpha=0.5, label="Observed data")
    plt.plot(x, true_line, linewidth=2, color="red", label="True regression line")

    plt.title("Observed Data with True Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)

    plt.show()


def plot_comparison(abc_parameters, exact_parameters, true_intercept, true_slope, output_path=None):
    """Compare ABC and exact posterior samples for intercept and slope."""
    abc_intercepts = [sample["intercept"] for sample in abc_parameters]
    abc_slopes = [sample["slope"] for sample in abc_parameters]

    plt.figure(figsize=(12, 5))

    # Intercept comparison
    plt.subplot(1, 2, 1)
    plt.hist(abc_intercepts, bins=30, density=True, alpha=0.5, label="ABC posterior")
    plt.hist(exact_parameters["intercept"], bins=30, density=True, alpha=0.5, label="Exact posterior")
    plt.axvline(true_intercept, color="red", linestyle="--", label="True intercept")
    plt.title("Posterior Comparison for Intercept")
    plt.xlabel("Intercept")
    plt.ylabel("Density")
    plt.legend()

    # Slope comparison
    plt.subplot(1, 2, 2)
    plt.hist(abc_slopes, bins=30, density=True, alpha=0.5, label="ABC posterior")
    plt.hist(exact_parameters["slope"], bins=30, density=True, alpha=0.5, label="Exact posterior")
    plt.axvline(true_slope, color="red", linestyle="--", label="True slope")
    plt.title("Posterior Comparison for Slope")
    plt.xlabel("Slope")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)

    plt.show()