from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_posterior(accepted_parameters, true_values=None, output_path=None):
    
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

def plot_observed_data(x, observed_data, true_intercept, true_slope, output_path=None):
    
    """
    Plot observed regression data and the true regression line.
    """
    x = np.asarray(x, dtype=float)
    observed_data = np.asarray(observed_data, dtype=float)

    if x.ndim != 1 or observed_data.ndim != 1:
        raise ValueError("x and observed_data must be one-dimensional.")

    if x.shape != observed_data.shape:
        raise ValueError(
            "x and observed_data must have the same shape."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")

    if not np.all(np.isfinite(observed_data)):
        raise ValueError(
            "observed_data must contain only finite values."
        )

    if not np.isfinite(true_intercept):
        raise ValueError("true_intercept must be finite.")

    if not np.isfinite(true_slope):
        raise ValueError("true_slope must be finite.")

    true_line = true_intercept + true_slope * x

    figure, axis = plt.subplots(figsize=(8, 6))

    axis.scatter(
        x,
        observed_data,
        alpha=0.5,
        label="Observed data",
    )

    axis.plot(
        x,
        true_line,
        linewidth=2,
        label="True regression line",
    )

    axis.set_title("Observed Data with True Regression Line")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend()

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(output_path)

    return figure

def plot_comparison(abc_parameters, exact_parameters, true_intercept, true_slope, output_path=None):
    """
    Compare ABC and exact posterior samples for intercept and slope.
    """
    if len(abc_parameters) == 0:
        raise ValueError(
            "At least one accepted ABC sample is required."
        )

    required_names = {"intercept", "slope"}

    for sample in abc_parameters:
        if not required_names.issubset(sample):
            raise ValueError(
                "Every ABC sample must contain "
                "'intercept' and 'slope'."
            )

    if not required_names.issubset(exact_parameters):
        raise ValueError(
            "exact_parameters must contain "
            "'intercept' and 'slope'."
        )

    abc_intercepts = np.asarray(
        [
            sample["intercept"]
            for sample in abc_parameters
        ],
        dtype=float,
    )

    abc_slopes = np.asarray(
        [
            sample["slope"]
            for sample in abc_parameters
        ],
        dtype=float,
    )

    exact_intercepts = np.asarray(
        exact_parameters["intercept"],
        dtype=float,
    )

    exact_slopes = np.asarray(
        exact_parameters["slope"],
        dtype=float,
    )

    arrays = {
        "ABC intercepts": abc_intercepts,
        "ABC slopes": abc_slopes,
        "exact intercepts": exact_intercepts,
        "exact slopes": exact_slopes,
    }

    for name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if values.size == 0:
            raise ValueError(
                f"{name} must not be empty."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} must contain only finite values."
            )

    if not np.isfinite(true_intercept):
        raise ValueError("true_intercept must be finite.")

    if not np.isfinite(true_slope):
        raise ValueError("true_slope must be finite.")

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    axes[0].hist(
        abc_intercepts,
        bins=30,
        density=True,
        alpha=0.5,
        label="ABC posterior",
    )

    axes[0].hist(
        exact_intercepts,
        bins=30,
        density=True,
        alpha=0.5,
        label="Exact posterior",
    )

    axes[0].axvline(
        true_intercept,
        linestyle="--",
        linewidth=2,
        label="True intercept",
    )

    axes[0].set_title(
        "Posterior Comparison for Intercept"
    )
    axes[0].set_xlabel("Intercept")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(
        abc_slopes,
        bins=30,
        density=True,
        alpha=0.5,
        label="ABC posterior",
    )

    axes[1].hist(
        exact_slopes,
        bins=30,
        density=True,
        alpha=0.5,
        label="Exact posterior",
    )

    axes[1].axvline(
        true_slope,
        linestyle="--",
        linewidth=2,
        label="True slope",
    )

    axes[1].set_title(
        "Posterior Comparison for Slope"
    )
    axes[1].set_xlabel("Slope")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        figure.savefig(output_path)

    return figure