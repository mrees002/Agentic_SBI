import inspect
import importlib.util
import numpy as np
import re

from distance import euclidean_distance
from inference import abc_function
from prior import UniformPrior
from summaries import mean_std_summary, regression_summary
from plots import plot_posterior


def inspect_simulator(file_path):

    """
    Returns the list of parameters of the simulator function defined in the given Python file.

    Parameters:
    file_path: str, path to the Python file containing the simulator function

    Returns:
    list: list of parameter names of the simulator function
    """

    # this loads a python file as a module from a file path
    spec = importlib.util.spec_from_file_location("simulator_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # get all functions defined in the module
    functions = inspect.getmembers(module, predicate=inspect.isfunction)

    # ask user to choose a function if there are multiple functions in the module
    if len(functions) == 0:
        raise ValueError(f"No functions found in {file_path}.")
    elif len(functions) == 1:
        func = functions[0][1]
    else:
        raise ValueError(f"More than one function found in {file_path}.")
    
    return func


def find_params(func):

    """
    Returns the parameters found in the simulator (assuming theta is a dictionary)

    Parameters:
    func: function, simulator function

    Returns:
    list: theta_keys, a list of the theta parameters
    dictionary: defaults, a dictionary of the parameters with default values
    list: required_inputs, a list of parameters with no set defaults
    """
    # create list of possible theta names
    THETA_NAMES = ["theta", "params", "parameters", "coefficients"]
    
    # get function signature
    sig = inspect.signature(func)
    
    defaults = {}
    theta_arg = None
    required_inputs = []
    
    # iterate through function parameters and assign values if necessary
    for name, param in sig.parameters.items():
        if name.lower() in THETA_NAMES:
            theta_arg = name
        elif param.default is not inspect.Parameter.empty:
            defaults[name] = param.default
        else:
            required_inputs.append(name)

    source = inspect.getsource(func)
    
    # Use regex to get theta parameters from function
    pattern = rf"{re.escape(theta_arg)}\[['\"](\w+)['\"]\]"
    theta_keys = re.findall(pattern, source)
    theta_keys = list(dict.fromkeys(theta_keys))
    
    return theta_keys, defaults, required_inputs


def get_user_input(theta_keys, defaults, required_inputs):

    """
    Gets user input for unkown parameters

    Parameters:
    theta_keys: list of parameters
    defaults: dictionary of parameters with values
    required_inputs: list of parameters without values

    Returns:
    dict: theta_input, a dictionary with theta parameters with lower and upper simulation bounds
    dict: param_input, a dictionary with parameters with values
    int: n_simulations, number of desired simulations of ABC
    float: epsilon, desired user epsilon
    int: summary_choice, an integer corresponding to the user's desired summary statistic

    """

    # show user what was found
    print("\nSimulator Analysis")
    print(f"Inferrable parameters found: {theta_keys}")
    print(f"Fixed settings found automatically: {defaults}")
    print(f"Need clarification on: {required_inputs}")

    # find bounds for priors
    theta_input = {}
    print("\nPrior Bounds")
    for name in theta_keys:
        lower = float(input(f"Enter lower bound for '{name}': "))
        upper = float(input(f"Enter upper bound for '{name}': "))
        theta_input[name] = (lower, upper)

    # clarify required inputs
    param_input = {**defaults}
    print("\nClarification Questions")
    for name in required_inputs:

        var_type = input(f"\nIs '{name}' a fixed setting or observed input data? (F/O): ")

        # ask user to fill in fixed setting
        if var_type.upper() == "F":
            param_input[name] = float(input(f"Enter value for '{name}': "))
        
        # ask user to fill in observed data
        elif var_type.upper() == "O":
            start = float(input(f"Enter start value for '{name}': "))
            end = float(input(f"Enter end value for '{name}': "))
            n_obs = int(input(f"Enter number of observations: "))
            param_input[name] = np.linspace(start, end, n_obs)


    # find simulations
    n_simulations = int(input("\nEnter number of simulations: "))

    # find epsilon
    epsilon = float(input("\nEnter epsilon value: "))

    # find summary choice
    print("\nAvailable summary functions:")
    print("1. mean_std_summary")
    print("2. regression_summary")
    summary_choice = int(input("Choose a summary function (1 or 2): "))
            
    return theta_input, param_input, n_simulations, epsilon, summary_choice

def synthetic_data(simulator, theta_input):

    """
    function that generates observed data based on a true theta entered by user

    Parameters:
    simulator: a simulator function
    theta_input: dictionary with theta parameters

    Returns:
    array: observed_data, an array of data generated by the true parameters
    true_theta: dictionary with parameters and true values
    """

    true_theta = {}
    print("\nEnter true parameter values to generate synthetic observed data:")
    
    # iterate through theta parameters to get true values
    for param_name in theta_input.keys():
        true_theta[param_name] = float(input(f"True value for {param_name}: "))
    
    # use simulator to generate data based on true theta
    observed_data = simulator(true_theta)

    return observed_data, true_theta

def run_agent(file_path):

    # get function
    func = inspect_simulator(file_path)

    # get parameters
    theta_keys, defaults, required_inputs = find_params(func)
    
    # get user input
    theta_input, param_input, n_simulations, epsilon, summary_choice = get_user_input(theta_keys, defaults, required_inputs)

    # build prior
    prior = UniformPrior(bounds=theta_input)

    # build simulator function
    def simulator(theta):
        # merge theta and param_input into a single dictionary
        return func(theta, **param_input)
    
    # ask for true parameter values and generate data
    observed_data, true_theta = synthetic_data(simulator, theta_input)

    # call summary function based on user choice
    def summary_fn(y):
        if summary_choice == 1:
            return mean_std_summary(y)
        elif summary_choice == 2:
            return regression_summary(param_input['x'], y)

    # run ABC inference
    print("\nRunning ABC inference...")
    accepted_parameters, accepted_distances = abc_function(prior, simulator, observed_data, summary_fn, euclidean_distance, epsilon, n_simulations)

    print(f"\nAccepted {len(accepted_parameters)} samples out of {n_simulations}")
    for param_name in theta_input.keys():
        values = [p[param_name] for p in accepted_parameters]
        print(f"Mean {param_name}: {np.mean(values):.3f}")

    plot_posterior(accepted_parameters, true_theta)


if __name__ == "__main__":
    run_agent("test_simulator.py")