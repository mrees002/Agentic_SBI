# config.py

import json


def create_config(
    output_path,
    inferred_parameters,
    fixed_inputs,
    priors,
    observed_data_path,
    epsilon,
    n_simulations,
    random_seed=123,
):
    config = {
        "inferred_parameters": inferred_parameters,
        "fixed_inputs": fixed_inputs,
        "priors": priors,
        "observed_data_path": observed_data_path,
        "abc": {
            "epsilon": epsilon,
            "n_simulations": n_simulations,
            "random_seed": random_seed,
        },
    }

    with open(output_path, "w") as file:
        json.dump(config, file, indent=4)

    return config