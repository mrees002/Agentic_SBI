import numpy as np
from pathlib import Path


def ask_inferred_parameters():
    text = input(
        "Enter inferred parameter names, "
        "separated by commas: "
    ).strip()

    names = [name.strip() for name in text.split(",")if name.strip()]

    if not names:
        raise ValueError(
            "At least one inferred parameter "
            "is required."
        )

    return names


def ask_prior_bounds(parameter_name):
    while True:
        lower = _ask_float(
            f"Lower prior bound for "
            f"{parameter_name}: "
        )

        upper = _ask_float(
            f"Upper prior bound for "
            f"{parameter_name}: "
        )

        if upper <= lower:
            print(
                "Upper bound must be greater "
                "than lower bound."
            )
            continue

        return lower, upper

def ask_random_seed(default=123):
    while True:
        text = input(
            f"Random seed [{default}]: "
        ).strip()

        if not text:
            return default

        try:
            random_seed = int(text)
        except ValueError:
            print(
                "Random seed must be an integer."
            )
            continue

        if random_seed < 0:
            print(
                "Random seed must be zero "
                "or greater."
            )
            continue

        return random_seed

def ask_observed_data_path():
    while True:
        path = input(
            "Path to observed data file: "
        ).strip()

        if not path:
            print(
                "An observed-data path "
                "is required."
            )
            continue

        if not path.lower().endswith(".npy"):
            print(
                "Observed data must be "
                "a .npy file."
            )
            continue

        try:
            data = np.load(
                path,
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            print(
                "Could not load observed data: "
                f"{error}"
            )
            continue

        if data.size == 0:
            print(
                "Observed-data array is empty."
            )
            continue

        if not np.all(np.isfinite(data)):
            print(
                "Observed data contains NaN "
                "or infinite values."
            )
            continue

        return path

def ask_epsilon():
    while True:
        value = _ask_float(
            "ABC epsilon: "
        )

        if value <= 0:
            print(
                "ABC epsilon must be greater "
                "than zero."
            )
            continue

        return value

def ask_n_simulations():
    while True:
        text = input(
            "Number of simulations: "
        ).strip()

        try:
            value = int(text)
        except ValueError:
            print(
                "Number of simulations must "
                "be an integer."
            )
            continue

        if value <= 0:
            print(
                "Number of simulations must "
                "be greater than zero."
            )
            continue

        return value

def ask_fixed_value(name):
    while True:
        print(f"\nFixed value for {name}:")
        print("  1. Enter a number")
        print("  2. Load a .npy file")
        print("  3. Generate a numeric sequence")

        selection = input(
            "Selection [1/2/3]: "
        ).strip()

        if selection == "1":
            return _ask_numeric_fixed_value(name)

        if selection == "2":
            return _ask_array_file(name)

        if selection == "3":
            return _ask_generated_array(name)

        print("Please enter 1, 2, or 3.")

def _ask_numeric_fixed_value(name):
    while True:
        text = input(
            f"Numeric value for {name}: "
        ).strip()

        try:
            value = float(text)
        except ValueError:
            print("Please enter a valid number.")
            continue

        return value, None

def _ask_array_file(name):
    while True:
        path = input(
            f"Path to .npy file for {name}: "
        ).strip()

        if not path:
            print("A file path is required.")
            continue

        if not path.lower().endswith(".npy"):
            print("The file must end in .npy.")
            continue

        try:
            value = np.load(
                path,
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            print(
                f"Could not load the array: {error}"
            )
            continue

        return value, path

def _ask_generated_array(name):
    while True:
        print(
            f"\nGenerate an evenly spaced "
            f"array for {name}."
        )

        start = _ask_float(
            "Start value: "
        )

        stop = _ask_float(
            "Stop value: "
        )

        number_of_points = (
            _ask_positive_integer(
                "Number of points: "
            )
        )

        if stop <= start:
            print(
                "Stop value must be greater "
                "than start value."
            )
            continue

        values = np.linspace(
            start,
            stop,
            number_of_points,
        )

        print(
            f"Generated {name} with shape "
            f"{values.shape}."
        )

        return values, None

def _ask_float(message):
    while True:
        text = input(message).strip()

        try:
            return float(text)
        except ValueError:
            print("Please enter a valid number.")

def _ask_positive_integer(message):
    while True:
        text = input(message).strip()

        try:
            value = int(text)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if value < 2:
            print(
                "Number of points must be "
                "at least 2."
            )
            continue

        return value


def display_analysis(analysis):
    print(
        "\nProposed simulator classification"
    )
    print(
        "---------------------------------"
    )

    print(
        f"RNG argument: "
        f"{analysis['rng_argument']}"
    )

    print(
        "Parameter container:",
        analysis["parameter_container"],
    )

    print("\nParameters inside container:")

    if analysis["container_parameters"]:
        for name in (
            analysis["container_parameters"]
        ):
            print(f"  {name}")
    else:
        print("  None")

    print("\nFixed values from defaults:")

    if analysis["fixed_values"]:
        for name, value in (
            analysis["fixed_values"].items()
        ):
            print(f"  {name} = {value}")
    else:
        print("  None")

    print(
        "\nFixed inputs requiring values:"
    )

    if (
        analysis[
            "fixed_inputs_without_values"
        ]
    ):
        for name in analysis[
            "fixed_inputs_without_values"
        ]:
            print(f"  {name}")
    else:
        print("  None")

    print(
        "\nDirect inferred parameters:"
    )

    if (
        analysis[
            "direct_inferred_parameters"
        ]
    ):
        for name in analysis[
            "direct_inferred_parameters"
        ]:
            print(f"  {name}")
    else:
        print("  None")

    if analysis["uncertain"]:
        print("\nWarnings:")

        for warning in analysis["uncertain"]:
            print(f"  {warning}")


def review_analysis(analysis):
    while True:
        display_analysis(analysis)

        response = input(
            "\nIs this classification "
            "correct? [y/n]: "
        ).strip().lower()

        if response in {"y", "yes"}:
            return analysis

        if response not in {"n", "no"}:
            print("Please enter y or n.")
            continue

        correct_analysis(analysis)


def correct_analysis(analysis):
    if (
        analysis["parameter_container"]
        is not None
    ):
        print(
            "\nThis simulator uses a parameter "
            "container."
        )
        print(
            "Under the current project scope, "
            "parameters inside the container are "
            "inferred and all other top-level "
            "arguments are fixed."
        )
        print(
            "Those roles cannot be moved without "
            "changing the simulator wrapper."
        )
        print(
            "Review the warnings or simulator "
            "definition if the proposal is wrong."
        )
        return

    movable_names = sorted(
        set(analysis["fixed_values"])
        | set(
            analysis[
                "fixed_inputs_without_values"
            ]
        )
        | set(
            analysis[
                "direct_inferred_parameters"
            ]
        )
    )

    if not movable_names:
        print(
            "There are no arguments available "
            "to move."
        )
        return

    print(
        "\nArguments that can be moved:"
    )

    for name in movable_names:
        category = get_analysis_category(
            analysis,
            name,
        )
        print(f"  {name}: {category}")

    name = input(
        "\nWhich argument is classified "
        "incorrectly? "
    ).strip()

    if name not in movable_names:
        print(
            f"{name!r} is not a movable "
            "top-level argument."
        )
        return

    print(
        "Current category:",
        get_analysis_category(
            analysis,
            name,
        ),
    )

    print("\nMove it to:")
    print(
        "  1. Fixed input requiring a value"
    )
    print("  2. Inferred parameter")

    choice = input(
        "Choose 1 or 2: "
    ).strip()

    if choice == "1":
        move_to_fixed_input(
            analysis,
            name,
        )
    elif choice == "2":
        move_to_inferred(
            analysis,
            name,
        )
    else:
        print("Invalid choice.")


def get_analysis_category(analysis, name):
    if name in analysis["fixed_values"]:
        return "fixed value from default"

    if name in analysis[
        "fixed_inputs_without_values"
    ]:
        return "fixed input requiring a value"

    if name in analysis[
        "direct_inferred_parameters"
    ]:
        return "direct inferred parameter"

    if name in analysis[
        "container_parameters"
    ]:
        return "parameter inside container"

    return None


def move_to_fixed_input(analysis, name):
    _remove_from_direct_role_lists(
        analysis,
        name,
    )

    analysis[
        "fixed_inputs_without_values"
    ].append(name)

    analysis[
        "fixed_inputs_without_values"
    ].sort()

    _refresh_inferred_parameters(
        analysis
    )

    analysis["evidence"][name] = (
        "Classified as a fixed input "
        "after user correction."
    )

    print(
        f"{name!r} moved to fixed inputs."
    )


def move_to_inferred(analysis, name):
    _remove_from_direct_role_lists(
        analysis,
        name,
    )

    analysis[
        "direct_inferred_parameters"
    ].append(name)

    analysis[
        "direct_inferred_parameters"
    ].sort()

    _refresh_inferred_parameters(
        analysis
    )

    analysis["evidence"][name] = (
        "Classified as directly inferred "
        "after user correction."
    )

    print(
        f"{name!r} moved to inferred "
        "parameters."
    )


def _remove_from_direct_role_lists(
    analysis,
    name,
):
    analysis["fixed_values"].pop(
        name,
        None,
    )

    categories = [
        "fixed_inputs_without_values",
        "direct_inferred_parameters",
        "unclassified",
    ]

    for category in categories:
        if name in analysis[category]:
            analysis[category].remove(name)


def _refresh_inferred_parameters(analysis):
    combined = (
        list(analysis["container_parameters"])
        + list(
            analysis[
                "direct_inferred_parameters"
            ]
        )
    )

    analysis["inferred_parameters"] = (
        list(dict.fromkeys(combined))
    )


def collect_missing_inputs(agent):
    missing = agent.get_missing_fields()

    if "inferred_parameters" in missing:
        names = ask_inferred_parameters()

        agent.set_inferred_parameters(
            *names
        )

    missing = agent.get_missing_fields()

    if "fixed_values" in missing:
        fixed_values = {}
        fixed_value_path = {}

        for name in missing["fixed_values"]:
            if name in agent.fixed_values:
                continue

            value, source_path = ask_fixed_value(
                name
            )

            fixed_values[name] = value

            if source_path is not None:
                fixed_value_path[name] = (
                    source_path
                )

        if fixed_values:
            agent.set_fixed_values(
                **fixed_values
            )

        agent.fixed_value_path.update(
            fixed_value_path
        )

    missing = agent.get_missing_fields()

    if "prior_bounds" in missing:
        bounds = {}

        for name in missing["prior_bounds"]:
            bounds[name] = ask_prior_bounds(
                name
            )

        agent.set_prior_bounds(
            **bounds
        )

    missing = agent.get_missing_fields()

    if "epsilon" in missing:
        agent.set_epsilon(
            ask_epsilon()
        )

    missing = agent.get_missing_fields()

    if "n_simulations" in missing:
        agent.set_n_sims(
            ask_n_simulations()
        )

    agent.set_random_seed(
        ask_random_seed(
            default=agent.random_seed
        )
    )

    generate_synthetic = (
        ask_generate_synthetic_data()
    )

    if generate_synthetic:
        true_values = {}

        for name in agent.inferred_parameters:
            lower, upper = (
                agent.prior_bounds[name]
            )

            while True:
                value = ask_true_parameter_value(
                    name
                )

                if lower <= value <= upper:
                    true_values[name] = value
                    break

                print(
                    f"True value for {name!r} "
                    f"must be between {lower} "
                    f"and {upper}."
                )

        agent.build_wrapper()

        agent.generate_synthetic_observed_data(
            true_values
        )

        print(
            "Synthetic observed data generated "
            f"with shape "
            f"{agent.observed_data.shape}."
        )

    else:
        missing = agent.get_missing_fields()

        if "observed_data_path" in missing:
            path = ask_observed_data_path()

            agent.load_observed_data(
                path
            )

    return agent

def ask_simulator_path():
    while True:
        path = input(
            "Path to simulator Python file: "
        ).strip()

        if not path:
            print(
                "A simulator file path "
                "is required."
            )
            continue

        simulator_path = Path(
            path
        ).expanduser()

        if simulator_path.suffix.lower() != ".py":
            print(
                "Simulator file must end "
                "in .py."
            )
            continue

        if not simulator_path.is_file():
            print(
                "Simulator file was not found: "
                f"{simulator_path}"
            )
            continue

        return str(simulator_path)

def ask_simulator_function_name():
    while True:
        name = input(
            "Simulator function name: "
        ).strip()

        if not name:
            print(
                "A simulator function name "
                "is required."
            )
            continue

        if not name.isidentifier():
            print(
                "Function name must be a valid "
                "Python identifier."
            )
            continue

        return name

def ask_use_config():
    while True:
        response = input(
            "Use an existing config file? [y/n]: "
        ).strip().lower()

        if response in {"y", "yes"}:
            return True

        if response in {"n", "no"}:
            return False

        print("Please enter y or n.")


def ask_config_path():
    while True:
        path = input(
            "Path to config file: "
        ).strip()

        if not path:
            print(
                "A config file path "
                "is required."
            )
            continue

        config_path = Path(
            path
        ).expanduser()

        if config_path.suffix.lower() != ".json":
            print(
                "Config file must end "
                "in .json."
            )
            continue

        if not config_path.is_file():
            print(
                "Config file was not found: "
                f"{config_path}"
            )
            continue

        return str(config_path)

def ask_generate_synthetic_data():
    while True:
        response = input(
            "Generate synthetic observed data? "
            "[y/n]: "
        ).strip().lower()

        if response in {"y", "yes"}:
            return True

        if response in {"n", "no"}:
            return False

        print("Please enter y or n.")

def ask_true_parameter_value(
    parameter_name,
):
    return _ask_float(
        f"True value for "
        f"{parameter_name}: "
    )