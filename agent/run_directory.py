from datetime import datetime
from pathlib import Path
import numpy as np


def create_run_directory(
    simulator_name,
    root_directory="runs",
):
    root_directory = Path(root_directory)

    root_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_simulator_name = _make_safe_name(
        simulator_name
    )

    date_text = datetime.now().strftime(
        "%Y%m%d"
    )

    run_directory = _find_available_directory(
        root_directory=root_directory,
        base_name=(
            f"{safe_simulator_name}_{date_text}"
        ),
    )

    run_directory.mkdir()

    return {
        "directory": run_directory,
        "config_path": (
            run_directory / "config.json"
        ),
        "results_path": (
            run_directory / "results.json"
        ),
        "posterior_path": (
            run_directory / "posterior.png"
        ),
    }


def _find_available_directory(
    root_directory,
    base_name,
):
    run_number = 1

    while True:
        directory_name = (
            f"{base_name}_{run_number:03d}"
        )

        candidate = (
            root_directory / directory_name
        )

        if not candidate.exists():
            return candidate

        run_number += 1


def _make_safe_name(name):
    safe_characters = []

    for character in name:
        if character.isalnum():
            safe_characters.append(
                character.lower()
            )
        else:
            safe_characters.append("_")

    safe_name = "".join(
        safe_characters
    )

    while "__" in safe_name:
        safe_name = safe_name.replace(
            "__",
            "_",
        )

    safe_name = safe_name.strip("_")

    if not safe_name:
        return "simulation"

    return safe_name

def save_run_data(
    agent,
    run_directory,
):
    run_directory = Path(run_directory)

    if agent.observed_data is None:
        raise ValueError(
            "Observed data is not available."
        )

    observed_file_name = "observed_data.npy"
    observed_path = (
        run_directory / observed_file_name
    )

    np.save(
        observed_path,
        np.asarray(agent.observed_data),
    )

    agent.observed_data_path = (
        observed_file_name
    )

    for name, value in agent.fixed_values.items():
        if not isinstance(value, np.ndarray):
            continue

        file_name = f"{name}.npy"
        output_path = (
            run_directory / file_name
        )

        np.save(
            output_path,
            value,
        )

        agent.fixed_value_path[name] = (
            file_name
        )

    return {
        "observed_data_path": observed_path,
        "fixed_value_paths": {
            name: run_directory / file_name
            for name, file_name
            in agent.fixed_value_path.items()
        },
    }