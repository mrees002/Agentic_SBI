from pathlib import Path
import importlib.util
import ast
import inspect
import textwrap
from collections.abc import Callable
from simulators import simulate_exponential_decay

import numpy as np

class SimulatorAgent:

    def __init__(self, simulator):
        
        # check if function is callable
        if not callable(simulator):
            raise TypeError("Simulator must be callable")

        self.simulator = simulator

        self.signature = None
        self.argument_names = []
        self.defaults = {}

        # Confirmed roles
        self.parameter_container = None
        self.inferred = []
        self.fixed = {}
        self.rng_argument = None
        self.unknown = []

        # Suggested roles
        self.possible_parameter_container = None
        self.possible_inferred = []
        self.possible_fixed = {}
        self.possible_rng = None

    def inspect_signature(self):
        
        # get function signature and argument names
        self.signature = inspect.signature(self.simulator)
        self.argument_names = list(self.signature.parameters)

        # initialize dictionary for defaults
        self.defaults = {}

        # iterate through signature and get defaults
        for name, parameter in self.signature.parameters.items():
            if parameter.default is not inspect.Parameter.empty:
                self.defaults[name] = parameter.default
    
    def identify_rng_argument(self):
        
        # create set of likely rng names
        rng_names = {"rng", "random_state", "random_generator", "generator"}

        # get a list of arguments that match rng names
        matches = [name for name in self.argument_names if name.lower() in rng_names]

        # set possible_rng to match if one matches
        if len(matches) == 1:
            self.possible_rng = matches[0]
            return self.possible_rng

        # raise error if multiple rng values exist
        if len(matches) > 1:
            print(f"Multiple possible RNG arguments found: {matches}")

        # get rng value if not found automatically
        while True:
            user_choice = input(
                "Enter the argument that receives the NumPy "
                "random generator, or enter 'none': "
            ).strip()

            # set possible rng equal to whatever the user inputs
            if user_choice in self.argument_names:
                self.possible_rng = user_choice
                return self.possible_rng

            # raise error if there is no rng input in the simulator
            if user_choice.lower() == "none":
                raise ValueError(
                    "The simulator must accept a NumPy random generator argument."
                )

            print(
                f"Invalid argument name. Choose from: "
                f"{self.argument_names}"
            )

    def sift_roles(self):

        # go through all remaining arguments that are not the rng
        remaining_arguments = [name for name in self.argument_names if name != self.possible_rng]

        for name in remaining_arguments:

            lower_name = name.lower()

            # if parameter is theta, let that be a container
            if lower_name == "theta":
                self.possible_parameter_container = name

            # set defaults automatically to begin
            elif name in self.defaults:
                self.possible_fixed[name] = self.defaults[name]

            # if not in either, add to unknowns
            else:
                self.unknown.append(name)

        return {
            "parameter_container": self.possible_parameter_container,
            "rng": self.possible_rng,
            "fixed": self.possible_fixed.copy(),
            "unknown": self.unknown.copy(),
        }
    
    def get_unknown_arguments(self):

        return self.unknown.copy()
    
    def unknown_classification(self, unknown_list):
        
        # iterate through unknown list
        for param in unknown_list:

            while True:
                
                # ask user where parameter should be inferred or fixed
                infer_fix_input = input(f"Should \"{param}\" be inferred or fixed? (I/F): ")
                
                if infer_fix_input == "I":
                    if param not in self.possible_inferred:
                        self.possible_inferred.append(param)
                    break

                if infer_fix_input == "F":
                    self.possible_fixed[param] = None
                    break

                print("Invalid input")

        self.unknown = [param for param in self.unknown if param not in unknown_list]

    def get_missing_fixed_values(self):

        return [name for name, value in self.possible_fixed.items() if value is None]

    def set_fixed_values(self, fixed_values):

        # ensure input is dictionary
        if not isinstance(fixed_values, dict):
            raise TypeError("fixed_values must be a dictionary.")

        # iterate through dictionary
        for name, value in fixed_values.items():

            # ensure value in dictionary is in the original arguments
            if name not in self.possible_fixed:
                raise ValueError(f'"{name}" is not classified as a fixed argument.')

            self.possible_fixed[name] = value

        # take note of values not entered
        missing = self.get_missing_fixed_values()

        if missing:
            print("Still missing fixed values:", missing)

    def validate_argument_roles(self):

        pass
                     

if __name__ == "__main__":

    pass