from datetime import datetime
from pathlib import Path
import shutil


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

def copy_config_to_run(
    source_config_path,
    destination_config_path,
):
    source_config_path = Path(
        source_config_path
    )

    destination_config_path = Path(
        destination_config_path
    )

    if not source_config_path.exists():
        raise FileNotFoundError(
            "Config file not found: "
            f"{source_config_path}"
        )

    shutil.copy2(
        source_config_path,
        destination_config_path,
    )

    return destination_config_path