import ast
import inspect
import textwrap

from agent.config import create_config, load_config
from prior import UniformPrior
from distance import euclidean_distance
from summaries import mean_std_summary

import numpy as np

class SimulatorAgent:

    def __init__(self, simulator):

        if not callable(simulator):
            raise TypeError("simulator must be callable.")

        self.simulator = simulator
        self.signature = inspect.signature(self.simulator)

        self.arguments = list(self.signature.parameters)
        self.rng_argument = None
        self.inferred_parameters = []
        self.parameter_container = None

        self.observed_data_path = None
        self.random_seed = 123

        self.fixed_values = {}
        self.prior_bounds = {}

        self.observed_data = None
        self.config = None
        self.wrapper = None

        # abc settings
        self.epsilon = None
        self.n_simulations = None

        # pre-set known functions
        self.prior = None # to be corrected with uniform prior after
        self.distance_fn = euclidean_distance
        self.summary_fn = mean_std_summary
    
    def load_observed_data(self, path):
        self.observed_data_path = str(path)
        self.observed_data = np.load(path)
        return self.observed_data
    
    def set_epsilon(self, epsilon):
        if not isinstance(epsilon, (float, int, np.number)):
            raise ValueError(f"epsilon must be a number.")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be greater than 0.")

        self.epsilon = epsilon
        return self.epsilon

    def set_n_sims(self, n_sims):
        if not isinstance(n_sims, int):
            raise ValueError(f"n_sims must be an integer.")
        if n_sims <= 0:
            raise ValueError(f"n_sims must be greater than 0.")

        self.n_simulations = n_sims
        return self.n_simulations
    
    def set_parameter_container(self, param_container):
        if param_container not in self.arguments:
            raise ValueError(f"{param_container} not in function arguments.")
        
        self.parameter_container = param_container
        return self.parameter_container
    
    def set_rng_argument(self, rng_argument):
        if rng_argument not in self.arguments:
            raise ValueError(f"{rng_argument} not in function arguments.")

        self.rng_argument = rng_argument
        return self.rng_argument
    
    def set_fixed_values(self, **values):
        new_fixed_values = {}

        for name, value in values.items():
            if name not in self.arguments:
                raise ValueError(f"{name} not in function arguments.")
            if name in self.inferred_parameters:
                raise ValueError(f"{name} is already in inferred parameters.")
            new_fixed_values[name] = value

        self.fixed_values = new_fixed_values

        return self.fixed_values

    def set_inferred_parameters(self, *parameter_names):
        new_inferred_parameters = []

        if self.parameter_container is not None:
            for param in parameter_names:
                new_inferred_parameters.append(param)

        else:
            for param in parameter_names:
                if param not in self.arguments:
                    raise ValueError(f"{param} not in function arguments.")
                
                else:
                    new_inferred_parameters.append(param)

        self.inferred_parameters = new_inferred_parameters
        return self.inferred_parameters

    def set_prior_bounds(self, **values):
        new_prior_bounds = {}

        for name, value in values.items():
            if name not in self.inferred_parameters:
                raise ValueError(f"{name} not in inferred parameters.")
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError(f"{name} must have bounds (lower, upper).")
            if value[1] <= value[0]:
                raise ValueError(f"{value} has improper bounds.")
            
            new_prior_bounds[name] = value

        if set(new_prior_bounds.keys()) != set(self.inferred_parameters):
            raise ValueError(f"Missing entries.")

        self.prior_bounds = new_prior_bounds
        self.prior = UniformPrior(self.prior_bounds)

    def create_config(self, output_path):
        if not self.inferred_parameters:
            raise ValueError("Inferred parameters have not been set.")
        if not self.prior_bounds:
            raise ValueError("Prior bounds have not been set.")
        if self.epsilon is None:
            raise ValueError("ABC epsilon has not been set.")
        if self.n_simulations is None:
            raise ValueError("Number of simulations has not been set.")
        if self.observed_data_path is None:
            raise ValueError("Observed data path has not been set.")

        self.config = create_config(
            output_path=output_path,
            simulator_name=self.simulator.__name__,
            parameter_container=self.parameter_container,
            rng_argument=self.rng_argument,
            inferred_parameters=self.inferred_parameters,
            fixed_values=self.fixed_values,
            prior_bounds=self.prior_bounds,
            observed_data_path=self.observed_data_path,
            epsilon=self.epsilon,
            n_simulations=self.n_simulations,
            random_seed=self.random_seed,
            summary_name=self.summary_fn.__name__,
            distance_name=self.distance_fn.__name__,
        )

        return self.config

    def configure_from_file(self, config_path):
        raw_config, settings = load_config(config_path)

        self.parameter_container = settings["parameter_container"]
        self.rng_argument = settings["rng_argument"]
        self.inferred_parameters = settings["inferred_parameters"]
        self.prior_bounds = settings["prior_bounds"]
        self.fixed_values = settings["fixed_values"]

        self.epsilon = settings["epsilon"]
        self.n_simulations = settings["n_simulations"]
        self.random_seed = settings["random_seed"]

        observed_data_path = settings["observed_data_path"]

        if observed_data_path is not None:
            self.load_observed_data(observed_data_path)

        self.prior = UniformPrior(self.prior_bounds)
        self.config = raw_config

        return self.config

    def build_wrapper(self): # requires that all or no inferred parameters are in theta
        if self.parameter_container is not None:

            def wrapper(theta, rng):
                arguments = {self.parameter_container: theta, **self.fixed_values}

                if self.rng_argument is not None:
                    arguments[self.rng_argument] = rng

                return self.simulator(**arguments)

        else:

            def wrapper(theta, rng):
                arguments = {}

                for name in self.inferred_parameters:
                    arguments[name] = theta[name]

                arguments.update(self.fixed_values)

                if self.rng_argument is not None:
                    arguments[self.rng_argument] = rng

                return self.simulator(**arguments)

        self.wrapper = wrapper
        return self.wrapper

    def test_abc(self):
        pass

    def run_abc(self):
        pass