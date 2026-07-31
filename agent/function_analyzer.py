import copy
import inspect

RNG_NAMES = {"rng","random_state", "generator"}
DATA_INPUT_NAMES = {"x", "xs", "t", "time", "times", "grid", "inputs",
    "input_data", "covariates", "features", "design_matrix",}

def analyze_agent(agent):

    simulator = agent.simulator
    signature = inspect.signature(simulator)
    parameters = signature.parameters

    _reject_unsupported_signature(parameters)

    analysis = {
        "arguments": list(parameters),
        "rng_argument": None,
        "fixed_values": {},
        "fixed_inputs_without_values": [],
        "direct_inferred_parameters": [],
        "inferred_parameters": [],
        "unclassified": [],
        "uncertain": [],
        "evidence": {},
    }

    _detect_rng(parameters, analysis)
    _collect_default_values(parameters, analysis)
    _collect_unclassified(parameters, analysis)
    _detect_data_inputs(parameters, analysis)
    _classify_remaining_as_direct_inferred(analysis)
    _refresh_inferred_parameters(analysis)

    return analysis

def _reject_unsupported_signature(parameters):
    unsupported = []

    for name, parameter in parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            unsupported.append(name)

    if unsupported:
        raise ValueError(f"Simulator signatures using *args or **kwargs are not supported: {unsupported}")

def _detect_rng(parameters, analysis):
    matches = []

    for name in parameters:
        if _looks_like_rng(name):
            matches.append(name)

    if not matches:
        return

    analysis["rng_argument"] = matches[0]
    analysis["evidence"][matches[0]] = ("Classified as RNG from its name or annotation.")

    if len(matches) > 1:
        analysis["uncertain"].append(f"Multiple possible RNG arguments were found: {matches}. Using {matches[0]!r}.")

def _looks_like_rng(name):
    return name.lower() in RNG_NAMES

def _collect_default_values(parameters, analysis):
    excluded = {analysis["rng_argument"]}

    for name, parameter in parameters.items():
        if name in excluded:
            continue

        if parameter.default is inspect.Parameter.empty:
            continue

        analysis["fixed_values"][name] = (copy.deepcopy(parameter.default))
        analysis["evidence"][name] = ("Classified as fixed because the function provides a default value.")

def _collect_unclassified(parameters, analysis):
    assigned_top_level = {
        analysis["rng_argument"],
        *analysis["fixed_values"],
    }

    analysis["unclassified"] = [name for name in parameters if name not in assigned_top_level]

def _detect_data_inputs(parameters, analysis):
    detected = []

    for name in analysis["unclassified"]:
        if name.lower() in DATA_INPUT_NAMES:
            detected.append(name)

            analysis["evidence"][name] = (
                "Suggested as fixed input because its name "
                "matches a recognized data-input name."
            )

    analysis["fixed_inputs_without_values"].extend(
        sorted(detected)
    )

    analysis["unclassified"] = [
        name
        for name in analysis["unclassified"]
        if name not in detected
    ]

def _classify_remaining_as_direct_inferred(analysis):
    remaining = list(analysis["unclassified"])
    analysis["direct_inferred_parameters"].extend(remaining)

    for name in remaining:
        analysis["evidence"][name] = (
            "Classified as inferred because it "
            "has no default and was not identified "
            "as RNG or fixed input data."
        )

    analysis["unclassified"] = []

def _refresh_inferred_parameters(analysis):
    analysis["inferred_parameters"] = list(analysis["direct_inferred_parameters"])

def validate_analysis(analysis):
    errors = []

    arguments = set(analysis["arguments"])

    rng_argument = analysis["rng_argument"]
    fixed_with_values = set(analysis["fixed_values"])
    fixed_without_values = set(analysis["fixed_inputs_without_values"])
    direct_inferred = set(analysis["direct_inferred_parameters"])
    combined_inferred = set(analysis["inferred_parameters"])
    unclassified = set(analysis["unclassified"])

    if rng_argument is not None:
        if rng_argument not in arguments:
            errors.append(f"RNG argument {rng_argument!r} is not in the simulator signature.")

    top_level_role_sets = {
        "fixed values": fixed_with_values,
        "fixed inputs": fixed_without_values,
        "direct inferred parameters": (direct_inferred),
    }

    role_names = list(top_level_role_sets)

    for index, first_name in enumerate(role_names):
        for second_name in role_names[index + 1:]:
            overlap = (top_level_role_sets[first_name] & top_level_role_sets[second_name])

            if overlap:
                errors.append(f"Arguments {sorted(overlap)} appear in both {first_name} and {second_name}.")

    for category, names in (top_level_role_sets.items()):
        invalid = names - arguments

        if invalid:
            errors.append(f"{category} contains names not in the function signature: {sorted(invalid)}")

    special_arguments = {name for name in {rng_argument} if name is not None}
    assigned_top_level = (fixed_with_values | fixed_without_values | direct_inferred | special_arguments)
    missing_top_level = (arguments - assigned_top_level)

    if missing_top_level:
        errors.append(f"Top-level arguments have no role: {sorted(missing_top_level)}")

    expected_combined = direct_inferred

    if combined_inferred != expected_combined:
        errors.append("The combined inferred parameter list does not match the container and direct inferred parameter lists.")

    if unclassified:
        errors.append(f"Unclassified arguments remain: {sorted(unclassified)}")

    if errors:
        raise ValueError("Invalid simulator classification:\n- " + "\n- ".join(errors))

    return True

def apply_analysis(agent, analysis):
    validate_analysis(analysis)

    if analysis["rng_argument"] is not None:
        agent.set_rng_argument(analysis["rng_argument"])

    agent.set_inferred_parameters(*analysis["inferred_parameters"])

    if analysis["fixed_values"]:
        agent.set_fixed_values(**analysis["fixed_values"])

    return agent