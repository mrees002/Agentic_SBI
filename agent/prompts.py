import numpy as np


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
    lower = float(
        input(
            f"Lower prior bound for "
            f"{parameter_name}: "
        )
    )

    upper = float(
        input(
            f"Upper prior bound for "
            f"{parameter_name}: "
        )
    )

    return lower, upper


def ask_observed_data_path():
    return input("Path to observed data file: ").strip()

def ask_epsilon():
    return float(input("ABC epsilon: "))

def ask_n_simulations():
    return int(input("Number of simulations: "))

def ask_fixed_value(parameter_name):
    text = input(
        f"Fixed value for {parameter_name} "
        "(number or .npy path): "
    ).strip()

    try:
        return float(text), None
    except ValueError:
        pass

    try:
        value = np.load(
            text,
            allow_pickle=False,
        )
    except (
        FileNotFoundError,
        IsADirectoryError,
        PermissionError,
        ValueError,
        OSError,
    ) as error:
        raise ValueError(
            f"{parameter_name} must be a number "
            "or a valid path to a NumPy .npy file."
        ) from error

    return value, text


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
                fixed_value_path[name] = source_path

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

        for name in missing[
            "prior_bounds"
        ]:
            bounds[name] = (
                ask_prior_bounds(name)
            )

        agent.set_prior_bounds(**bounds)

    missing = agent.get_missing_fields()

    if "observed_data_path" in missing:
        path = ask_observed_data_path()
        agent.load_observed_data(path)

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

    return agent