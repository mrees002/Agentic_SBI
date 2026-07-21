import json
from pathlib import Path

import numpy as np


def make_json_safe(value):

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    return value


def create_config(
    output_path,
    simulator_name,
    parameter_container,
    rng_argument,
    inferred_parameters,
    fixed_values,
    fixed_value_path,
    prior_bounds,
    observed_data_path,
    epsilon,
    n_simulations,
    random_seed,
    summary_name,
    distance_name,
):
    config = {
        "simulator": {
            "name": simulator_name,
            "parameter_container": parameter_container,
            "rng_argument": rng_argument,
        },
        "inference": {
            "inferred_parameters": inferred_parameters,
            "prior_bounds": prior_bounds,
        },
        "fixed_values": fixed_values,
        "fixed_value_path": fixed_value_path,
        "observed_data_path": observed_data_path,
        "abc": {
            "epsilon": epsilon,
            "n_simulations": n_simulations,
            "random_seed": random_seed,
        },
        "functions": {
            "summary": summary_name,
            "distance": distance_name,
        },
    }

    config = make_json_safe(config)

    output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    return config


def load_config_file(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    if config_path.suffix != ".json":
        raise ValueError(
            "Config file must be a JSON file."
        )

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_config(config):
    simulator_config = config["simulator"]
    inference_config = config["inference"]
    abc_config = config["abc"]

    fixed_values = {
        name: np.asarray(value)
        if isinstance(value, list)
        else value
        for name, value in config.get(
            "fixed_values",
            {}
        ).items()
    }

    for name, path in config.get(
        "fixed_value_path",
        {}
    ).items():
        fixed_values[name] = np.load(
            path,
            allow_pickle=False,
        )

    normalized = {
        "parameter_container": simulator_config.get(
            "parameter_container"
        ),
        "rng_argument": simulator_config.get(
            "rng_argument"
        ),
        "inferred_parameters": list(
            inference_config[
                "inferred_parameters"
            ]
        ),
        "prior_bounds": {
            name: tuple(bounds)
            for name, bounds in inference_config[
                "prior_bounds"
            ].items()
        },
        "fixed_values": fixed_values,
        "observed_data_path": config.get(
            "observed_data_path"
        ),
        "epsilon": abc_config["epsilon"],
        "n_simulations": abc_config[
            "n_simulations"
        ],
        "random_seed": abc_config[
            "random_seed"
        ],
    }

    return normalized

def validate_config(config):
    required_sections = {
        "simulator",
        "inference",
        "abc",
    }

    missing_sections = required_sections - set(config)

    if missing_sections:
        raise ValueError(
            f"Missing config sections: {sorted(missing_sections)}"
        )

    required_abc_fields = {
        "epsilon",
        "n_simulations",
        "random_seed",
    }

    missing_abc_fields = (
        required_abc_fields
        - set(config["abc"])
    )

    if missing_abc_fields:
        raise ValueError(
            f"Missing ABC settings: {sorted(missing_abc_fields)}"
        )


def load_config(config_path):
    raw_config = load_config_file(config_path)
    validate_config(raw_config)
    normalized_config = normalize_config(raw_config)

    return raw_config, normalized_config