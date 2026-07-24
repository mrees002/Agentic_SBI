import json
from pathlib import Path

import numpy as np


def create_results_summary(
    agent,
    config_path,
):
    config_path = Path(config_path)

    _validate_completed_run(agent)

    accepted_count = len(
        agent.accepted_parameters
    )

    return {
        "config": {
            "file": config_path.name,
        },
        "simulator": {
            "name": agent.simulator.__name__,
        },
        "abc_results": {
            "accepted_samples": accepted_count,
            "acceptance_rate": (
                accepted_count
                / agent.n_simulations
            ),
        },
        "parameters": _summarize_parameters(
            agent.accepted_parameters,
            agent.inferred_parameters,
        ),
        "distances": _summarize_distances(
            agent.accepted_distances,
        ),
    }

def save_results(
    agent,
    config_path,
    output_path=None,
):
    config_path = Path(config_path)

    if output_path is None:
        output_path = get_results_path(
            config_path
        )
    else:
        output_path = Path(output_path)

    results = create_results_summary(
        agent,
        config_path,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(results),
            file,
            indent=4,
        )

    return results, output_path


def _validate_completed_run(agent):
    if agent.n_simulations is None:
        raise ValueError(
            "Number of simulations is not configured."
        )

    if agent.n_simulations <= 0:
        raise ValueError(
            "Number of simulations must be greater than zero."
        )

    if not agent.accepted_parameters:
        raise ValueError(
            "No accepted parameter samples are available. "
            "Run ABC before creating results."
        )

    if len(agent.accepted_parameters) != len(
        agent.accepted_distances
    ):
        raise ValueError(
            "Accepted parameter and distance counts "
            "do not match."
        )


def _summarize_parameters(
    accepted_parameters,
    parameter_names,
):
    summary = {}

    for parameter_name in parameter_names:
        values = _extract_parameter_values(
            accepted_parameters,
            parameter_name,
        )

        summary[parameter_name] = {
            "mean": np.mean(values),
            "standard_deviation": np.std(
                values,
                ddof=0,
            ),
            "median": np.median(values),
            "minimum": np.min(values),
            "maximum": np.max(values),
            "credible_interval_95": {
                "lower": np.percentile(
                    values,
                    2.5,
                ),
                "upper": np.percentile(
                    values,
                    97.5,
                ),
            },
        }

    return summary


def _extract_parameter_values(
    accepted_parameters,
    parameter_name,
):
    values = []

    for index, sample in enumerate(
        accepted_parameters
    ):
        if not isinstance(sample, dict):
            raise TypeError(
                "Each accepted parameter sample "
                "must be a dictionary. "
                f"Sample {index} has type "
                f"{type(sample).__name__}."
            )

        if parameter_name not in sample:
            raise ValueError(
                f"Accepted sample {index} does not "
                f"contain parameter "
                f"{parameter_name!r}."
            )

        value = sample[parameter_name]

        if not isinstance(
            value,
            (int, float, np.number),
        ):
            raise TypeError(
                f"Accepted value for "
                f"{parameter_name!r} must be "
                "numeric."
            )

        values.append(float(value))

    return np.asarray(
        values,
        dtype=float,
    )


def _summarize_distances(
    accepted_distances,
):
    distances = np.asarray(
        accepted_distances,
        dtype=float,
    )

    if distances.ndim != 1:
        distances = distances.reshape(-1)

    if not np.all(np.isfinite(distances)):
        raise ValueError(
            "Accepted distances contain NaN "
            "or infinite values."
        )

    return {
        "minimum": np.min(distances),
        "mean": np.mean(distances),
        "median": np.median(distances),
        "maximum": np.max(distances),
        "standard_deviation": np.std(
            distances,
            ddof=0,
        ),
    }

def get_results_path(config_path):
    config_path = Path(config_path)

    name = config_path.stem

    if name.endswith("_config"):
        name = name.removesuffix("_config")

    return config_path.with_name(
        f"{name}_results.json"
    )


def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value