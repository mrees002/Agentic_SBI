from datetime import datetime
from pathlib import Path
import shutil


def create_run_directory(
    config_path,
    simulator_name,
    root_directory="runs",
):
    """
    Create a unique directory for one inference run.

    The source config is copied into the directory
    as config.json.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"Config path is not a file: {config_path}"
        )

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

    copied_config_path = (
        run_directory / "config.json"
    )

    shutil.copy2(
        config_path,
        copied_config_path,
    )

    return {
        "directory": run_directory,
        "config_path": copied_config_path,
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