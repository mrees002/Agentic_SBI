import matplotlib.pyplot as plt

def plot_posterior(accepted_parameters, true_values):
    """
    Plots posterior histograms for any set of parameters.
    
    Parameters:
    accepted_parameters: list of accepted parameter dictionaries from ABC
    true_values: dictionary of true parameter values e.g. {"intercept": 1.0, "slope": 2.0}
    """
    parameter_names = list(true_values.keys())
    n_params = len(parameter_names)

    plt.figure(figsize=(6 * n_params, 5))

    for i, name in enumerate(parameter_names):
        values = [p[name] for p in accepted_parameters]
        plt.subplot(1, n_params, i + 1)
        plt.hist(values, bins=30, density=True, alpha=0.7, color='blue')
        plt.axvline(true_values[name], color='red', linestyle='dashed', 
                    linewidth=2, label=f'True {name}')
        plt.title(f'Posterior: {name}')
        plt.xlabel(name)
        plt.ylabel('Density')
        plt.legend()

    plt.tight_layout()
    plt.savefig("results/posterior.png")
    plt.show()

def plot_observed_data(x, observed_data, true_intercept, true_slope):
    """
    Plots the observed data along with the true regression line.
    
    Parameters:
    x: array of x values
    observed_data: array of observed y values
    true_intercept: true intercept value
    true_slope: true slope value
    """

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
    """
    Plots the comparison of ABC and exact posterior distributions for intercept and slope.

    Parameters:
    abc_parameters: list of accepted parameter dictionaries from ABC
    exact_parameters: dictionary containing exact posterior samples for intercept and slope
    true_intercept: true intercept value to mark on plot
    true_slope: true slope value to mark on plot
    """

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