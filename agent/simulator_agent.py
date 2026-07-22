import inspect

from agent.config import create_config, load_config
from prior import UniformPrior
from distance import euclidean_distance
from summaries import mean_std_summary
from inference import abc_function
from plots import plot_posterior

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
        self.fixed_values = {}
        self.prior_bounds = {}

        self.observed_data = None
        self.config = None
        self.wrapper = None

        # add true parameter values if needed
        self.true_parameter_values = None

        # add paths
        self.fixed_value_path = {}
        self.simulator_path = None
        self.observed_data_path = None

        # abc settings
        self.epsilon = None
        self.n_simulations = None
        self.random_seed = 123
        self.abc_function = abc_function

        # pre-set known functions
        self.prior = None # to be corrected with uniform prior after
        self.distance_fn = euclidean_distance
        self.summary_fn = mean_std_summary

        # save results
        self.accepted_distances = []
        self.accepted_parameters = []
    
    def load_observed_data(self, path):
        self.observed_data_path = str(path)
        self.observed_data = np.load(path)
        return self.observed_data
    
    def set_simulator_path(self, path):
        self.simulator_path = path
        return self.simulator_path
    
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
    
    def set_random_seed(self, random_seed):
        if not isinstance(random_seed, int):
            raise ValueError("random_seed must be an integer.")
        if random_seed < 0:
            raise ValueError("random_seed must be zero or greater.")

        self.random_seed = random_seed
        return self.random_seed
    
    def set_fixed_values(self, **values):
        new_fixed_values = {}

        for name, value in values.items():
            if name not in self.arguments:
                raise ValueError(f"{name} not in function arguments.")
            if (self.parameter_container is None and name in self.inferred_parameters):
                raise ValueError(f"{name} is already an inferred parameter.")

            new_fixed_values[name] = value
        self.fixed_values.update(new_fixed_values)

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
        
        literal_fixed_values = {name: value for name, value in self.fixed_values.items()
                                if name not in self.fixed_value_path}

        self.config = create_config(
            output_path=output_path,
            simulator_name=self.simulator.__name__,
            simulator_path=self.simulator_path,
            parameter_container=self.parameter_container,
            rng_argument=self.rng_argument,
            inferred_parameters=self.inferred_parameters,
            fixed_values=literal_fixed_values,
            fixed_value_path = self.fixed_value_path,
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
        if self.observed_data is None:
            raise ValueError("Observed data has not been loaded.")

        if self.prior is None:
            raise ValueError("Prior bounds have not been set.")

        if self.wrapper is None:
            raise ValueError("Simulator wrapper has not been built.")

        rng = np.random.default_rng(self.random_seed)

        # 1. Sample parameters from the configured prior
        try:
            test_theta = self.prior.sample(rng)
        except Exception as error:
            raise RuntimeError(f"Prior sampling failed: {error}") from error

        # 2. Execute the simulator
        try:
            simulated_data = self.wrapper(test_theta, rng)
        except Exception as error:
            raise RuntimeError(f"Simulator execution failed: {error}") from error

        simulated_data = np.asarray(simulated_data)
        observed_data = np.asarray(self.observed_data)

        # 3. Validate simulator output
        if simulated_data.shape != observed_data.shape:
            raise ValueError(
                "Simulator output is incompatible with observed data: "
                f"simulated shape {simulated_data.shape}, "
                f"observed shape {observed_data.shape}."
            )
        
        # 4. Run a very small ABC smoke test with automatic acceptance
        try:
            accepted_parameters, accepted_distances = self.abc_function(
                prior=self.prior,
                simulator=self.wrapper,
                observed_data=self.observed_data,
                summary_fn=self.summary_fn,
                distance_fn=self.distance_fn,
                epsilon=np.inf,
                n_simulations=2,
                rng=rng,
            )
        except Exception as error:
            raise RuntimeError(f"ABC pipeline test failed: {error}") from error

        if len(accepted_parameters) == 0:
            raise ValueError("ABC test produced no accepted parameter samples.")

        if len(accepted_parameters) != len(accepted_distances):
            raise ValueError(
                "ABC returned inconsistent parameter and distance counts."
            )

        accepted_distances = np.asarray(accepted_distances)

        if not np.all(np.isfinite(accepted_distances)):
            raise ValueError("ABC returned non-finite distances.")

        return {
            "success": True,
            "sampled_parameters": test_theta,
            "simulated_shape": simulated_data.shape,
            "observed_shape": observed_data.shape,
            "accepted_samples": len(accepted_parameters),
        }

    def run_abc(self):
        if self.observed_data is None:
            raise ValueError("Observed data has not been loaded.")

        if self.prior is None:
            raise ValueError("Prior bounds have not been set.")

        if self.wrapper is None:
            raise ValueError("Simulator wrapper has not been built.")

        if self.epsilon is None:
            raise ValueError("ABC epsilon has not been set.")

        if self.n_simulations is None:
            raise ValueError("Number of simulations has not been set.")

        rng = np.random.default_rng(self.random_seed)

        try:
            accepted_parameters, accepted_distances = self.abc_function(
                prior=self.prior,
                simulator=self.wrapper,
                observed_data=self.observed_data,
                summary_fn=self.summary_fn,
                distance_fn=self.distance_fn,
                epsilon=self.epsilon,
                n_simulations=self.n_simulations,
                rng=rng,
            )
        except Exception as error:
            raise RuntimeError(f"ABC inference failed: {error}") from error

        if len(accepted_parameters) == 0:
            raise ValueError(
                "ABC inference completed but accepted no parameter samples. "
                "Consider increasing epsilon or the simulation budget."
            )

        if len(accepted_parameters) != len(accepted_distances):
            raise ValueError("ABC returned inconsistent parameter and distance counts.")

        accepted_distances = np.asarray(accepted_distances, dtype=float,)

        if not np.all(np.isfinite(accepted_distances)):
            raise ValueError("ABC returned NaN or infinite distances.")
        
        self.accepted_parameters = accepted_parameters
        self.accepted_distances = accepted_distances

        return accepted_parameters, accepted_distances
    
    def plot_posterior_hist(self, output_path = None):

        if not self.accepted_parameters:
            raise ValueError("No accepted parameters.")
        
        plot_posterior(self.accepted_parameters, output_path = output_path)

    def get_missing_fields(self):
        missing = {}

        if not self.inferred_parameters:
            missing["inferred_parameters"] = True

        expected_fixed = set(self.arguments)

        if self.parameter_container is not None:
            expected_fixed.discard(self.parameter_container)

        if self.rng_argument is not None:
            expected_fixed.discard(self.rng_argument)

        expected_fixed -= set(self.inferred_parameters)

        missing_fixed = sorted(expected_fixed - set(self.fixed_values))

        if missing_fixed:
            missing["fixed_values"] = missing_fixed

        missing_priors = sorted(set(self.inferred_parameters) - set(self.prior_bounds))

        if missing_priors:
            missing["prior_bounds"] = missing_priors

        if self.epsilon is None:
            missing["epsilon"] = True

        if self.n_simulations is None:
            missing["n_simulations"] = True

        if self.observed_data is None:
            missing["observed_data_path"] = True

        return missing

    def generate_synthetic_observed_data(self, true_parameter_values):
        if self.wrapper is None:
            self.build_wrapper()

        missing_parameters = [name for name in self.inferred_parameters if name not in true_parameter_values]

        if missing_parameters:
            raise ValueError(
                "Missing true values for: "
                f"{missing_parameters}"
            )

        rng = np.random.default_rng(self.random_seed + 1)

        observed_data = self.wrapper(true_parameter_values, rng)

        observed_data = np.asarray(observed_data)

        if observed_data.size == 0:
            raise ValueError("Synthetic observed data is empty.")

        if not np.all(np.isfinite(observed_data)):
            raise ValueError(
                "Synthetic observed data contains "
                "NaN or infinite values."
            )

        self.observed_data = observed_data
        self.true_parameter_values = dict(true_parameter_values)

        return self.observed_data