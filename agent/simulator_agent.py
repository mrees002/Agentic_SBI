import ast
import inspect
import textwrap

from agent.config import create_config
from prior import UniformPrior

import numpy as np

class SimulatorAgent:

    def __init__(self, simulator):

        if not callable(simulator):
            raise TypeError("simulator must be callable.")

        self.simulator = simulator
        self.signature = None

        self.arguments = []
        self.inferred_parameters = []
        self.fixed_inputs = []
        self.rng_argument = None
        self.observed_data = None

        self.fixed_values = {}
        self.priors = {}

        self.config = None

    def inspect_signature(self):
        self.signature = inspect.signature(self.simulator)
        self.arguments = list(self.signature.parameters)

    def load_observed_data(self, path):
        self.observed_data = np.load(path)
        return self.observed_data

    def propose_argument_roles(self):

        self.inferred_parameters = []
        self.fixed_inputs = []
        self.rng_argument = None

        for name in self.arguments:
            lower_name = name.lower()

            if "rng" in lower_name:
                self.rng_argument = name

            elif lower_name in {"x", "time", "t"}:
                self.fixed_inputs.append(name)

            else:
                self.inferred_parameters.append(name)

        if self.rng_argument is None:
            raise ValueError("rng value not found.")

    def confirm_argument_roles(self, inferred_parameters, fixed_inputs):
        
        all_names = set(inferred_parameters + fixed_inputs)
        expected_names = set(self.arguments)

        if self.rng_argument is not None:
            expected_names.remove(self.rng_argument)

        if all_names != expected_names:
            raise ValueError("Every non-rng argument must be classified exactly once.")

        if set(inferred_parameters) & set(fixed_inputs):
            raise ValueError("An argument cannot be both inferred and fixed.")

        self.inferred_parameters = list(inferred_parameters)
        self.fixed_inputs = list(fixed_inputs)

    def set_fixed_values(self, **values):

        for name, value in values.items():

            if name not in self.fixed_inputs:
                raise ValueError(f"{name} is not a fixed input.")

            self.fixed_values[name] = value

    def set_uniform_priors(self, priors):

        for name in self.inferred_parameters:

            if name not in priors:
                raise ValueError(f"Missing prior for {name}.")

        self.priors = priors

    def set_theta_parameters(self, parameter_names):
        
        self.inferred_parameters = list(parameter_names)

    def build_wrapper(self):

        def wrapper(theta, rng):

            if "theta" in self.arguments:
                return self.simulator(theta, rng)

            arguments = {}

            for name in self.inferred_parameters:
                arguments[name] = theta[name]

            for name in self.fixed_inputs:
                arguments[name] = self.fixed_values[name]

            arguments[self.rng_argument] = rng

            return self.simulator(**arguments)

        self.wrapper = wrapper

        return self.wrapper

    def make_config(self, output_path, observed_data_path, epsilon, n_simulations):
        
        self.config = create_config(
            output_path=output_path,
            inferred_parameters=self.inferred_parameters,
            fixed_inputs=self.fixed_inputs,
            priors=self.priors,
            observed_data_path=observed_data_path,
            epsilon=epsilon,
            n_simulations=n_simulations,
        )

        return self.config

    def validate(self, observed_data, rng, abc_function, prior, summary_fn, distance_fn):

        # Run simulator once with random parameters
        theta = prior.sample(rng)

        try:
            simulated_data = self.wrapper(theta, rng)
        except Exception as error:
            raise RuntimeError(
                f"Simulator validation failed: {error}"
            ) from error

        observed_data = np.asarray(observed_data)
        simulated_data = np.asarray(simulated_data)

        # Check output compatibility
        if simulated_data.shape != observed_data.shape:
            raise ValueError(
                "Simulated data and observed data have different shapes."
            )

        if not np.all(np.isfinite(simulated_data)):
            raise ValueError(
                "Simulator returned non-finite values."
            )

        # Run a very small ABC test
        accepted_parameters, accepted_distances = abc_function(
            prior=prior,
            simulator=self.wrapper,
            observed_data=observed_data,
            summary_fn=summary_fn,
            distance_fn=distance_fn,
            epsilon=np.inf,
            n_simulations=2,
            rng=rng,
        )

        if len(accepted_parameters) == 0:
            raise ValueError(
                "ABC validation did not produce posterior samples."
            )

        return True
    
    def run_abc(self, observed_data, abc_function, summary_fn, distance_fn, epsilon, n_simulations, rng):

        prior = UniformPrior(self.priors)
        rng = np.random.default_rng()

        return abc_function(
            prior=prior,
            simulator=self.wrapper,
            observed_data=observed_data,
            summary_fn=summary_fn,
            distance_fn=distance_fn,
            epsilon=epsilon,
            n_simulations=n_simulations,
            rng=rng,
        )