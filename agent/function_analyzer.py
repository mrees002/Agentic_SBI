import ast
import copy
import inspect
import textwrap


RNG_NAMES = {
    "rng",
    "random_state",
    "generator",
}

PARAMETER_CONTAINER_NAMES = {
    "theta",
    "params",
    "parameters",
}

DATA_INPUT_NAMES = {
    "x",
    "xs",
    "t",
    "time",
    "times",
    "grid",
    "inputs",
    "input_data",
    "covariates",
    "features",
    "design_matrix",
}


def analyze_agent(agent):
    """
    Inspect the simulator and return a proposed classification.

    This function does not modify the agent.
    """
    simulator = agent.simulator
    signature = inspect.signature(simulator)
    parameters = signature.parameters

    _reject_unsupported_signature(parameters)

    analysis = {
        "arguments": list(parameters),
        "rng_argument": None,
        "parameter_container": None,
        "fixed_values": {},
        "fixed_inputs_without_values": [],
        "container_parameters": [],
        "direct_inferred_parameters": [],
        "inferred_parameters": [],
        "unclassified": [],
        "uncertain": [],
        "evidence": {},
    }

    _detect_rng(parameters, analysis)
    _detect_parameter_container(parameters, analysis)
    _collect_default_values(parameters, analysis)
    _inspect_parameter_container(simulator, analysis)
    _collect_unclassified(parameters, analysis)

    if analysis["parameter_container"] is not None:
        _classify_container_style_arguments(analysis)
    else:
        _detect_data_inputs(
            simulator,
            parameters,
            analysis,
        )
        _classify_remaining_as_direct_inferred(
            analysis
        )

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
        raise ValueError(
            "Simulator signatures using *args or **kwargs "
            f"are not supported: {unsupported}"
        )


def _detect_rng(parameters, analysis):
    matches = []

    for name, parameter in parameters.items():
        if _looks_like_rng(
            name,
            parameter.annotation,
        ):
            matches.append(name)

    if not matches:
        return

    analysis["rng_argument"] = matches[0]
    analysis["evidence"][matches[0]] = (
        "Classified as RNG from its name or annotation."
    )

    if len(matches) > 1:
        analysis["uncertain"].append(
            "Multiple possible RNG arguments were found: "
            f"{matches}. Using {matches[0]!r}."
        )


def _looks_like_rng(name, annotation):
    if name.lower() in RNG_NAMES:
        return True

    if annotation is inspect.Parameter.empty:
        return False

    annotation_text = str(annotation).lower()

    return (
        "generator" in annotation_text
        or "randomstate" in annotation_text
        or "random_state" in annotation_text
    )


def _detect_parameter_container(
    parameters,
    analysis,
):
    matches = [
        name
        for name in parameters
        if name.lower() in PARAMETER_CONTAINER_NAMES
    ]

    if not matches:
        return

    analysis["parameter_container"] = matches[0]
    analysis["evidence"][matches[0]] = (
        "Classified as a parameter container "
        "from its name."
    )

    if len(matches) > 1:
        analysis["uncertain"].append(
            "Multiple possible parameter containers "
            f"were found: {matches}. "
            f"Using {matches[0]!r}."
        )


def _collect_default_values(parameters, analysis):
    excluded = {
        analysis["rng_argument"],
        analysis["parameter_container"],
    }

    for name, parameter in parameters.items():
        if name in excluded:
            continue

        if parameter.default is inspect.Parameter.empty:
            continue

        analysis["fixed_values"][name] = (
            copy.deepcopy(parameter.default)
        )

        analysis["evidence"][name] = (
            "Classified as fixed because the "
            "function provides a default value."
        )


def _inspect_parameter_container(
    simulator,
    analysis,
):
    container_name = analysis["parameter_container"]

    if container_name is None:
        return

    tree = _get_source_tree(
        simulator,
        analysis,
        purpose="parameter container inspection",
    )

    if tree is None:
        analysis["uncertain"].append(
            f"The contents of {container_name!r} "
            "could not be determined."
        )
        return

    visitor = ParameterContainerVisitor(
        container_name
    )
    visitor.visit(tree)

    container_parameters = sorted(
        visitor.parameter_names
    )

    analysis["container_parameters"].extend(
        container_parameters
    )

    for name in container_parameters:
        analysis["evidence"][name] = (
            f"Found {container_name}[{name!r}] "
            "in the simulator source."
        )

    if not container_parameters:
        analysis["uncertain"].append(
            f"No named parameters were found inside "
            f"{container_name!r}."
        )


def _get_source_tree(
    simulator,
    analysis,
    purpose,
):
    try:
        source = inspect.getsource(simulator)
    except (OSError, TypeError):
        analysis["uncertain"].append(
            f"Source was unavailable for {purpose}."
        )
        return None

    source = textwrap.dedent(source)

    try:
        return ast.parse(source)
    except SyntaxError:
        analysis["uncertain"].append(
            f"Source could not be parsed for {purpose}."
        )
        return None


class ParameterContainerVisitor(ast.NodeVisitor):

    def __init__(self, container_name):
        self.container_name = container_name
        self.parameter_names = set()

    def visit_Subscript(self, node):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == self.container_name
        ):
            key = _extract_string_value(
                node.slice
            )

            if key is not None:
                self.parameter_names.add(key)

        self.generic_visit(node)

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(
                node.func.value,
                ast.Name,
            )
            and (
                node.func.value.id
                == self.container_name
            )
            and node.args
        ):
            key = _extract_string_value(
                node.args[0]
            )

            if key is not None:
                self.parameter_names.add(key)

        self.generic_visit(node)


def _extract_string_value(node):
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    ):
        return node.value

    return None


def _collect_unclassified(
    parameters,
    analysis,
):
    assigned_top_level = {
        analysis["rng_argument"],
        analysis["parameter_container"],
        *analysis["fixed_values"],
    }

    analysis["unclassified"] = [
        name
        for name in parameters
        if name not in assigned_top_level
    ]


def _classify_container_style_arguments(
    analysis,
):
    """
    Under the current wrapper design, all inferred
    parameters belong inside the parameter container.

    Remaining top-level arguments must therefore be
    treated as fixed inputs.
    """
    remaining = list(analysis["unclassified"])

    analysis[
        "fixed_inputs_without_values"
    ].extend(remaining)

    for name in remaining:
        analysis["evidence"][name] = (
            "Classified as a fixed top-level input "
            "because this simulator uses a parameter "
            "container for inferred parameters."
        )

    analysis["unclassified"] = []


def _detect_data_inputs(
    simulator,
    parameters,
    analysis,
):
    candidates = set(analysis["unclassified"])

    if not candidates:
        return

    detected = set()

    for name in candidates:
        parameter = parameters[name]

        if _annotation_suggests_data(
            parameter.annotation
        ):
            detected.add(name)
            analysis["evidence"][name] = (
                "Classified as fixed because its "
                "annotation suggests array-like data."
            )

    source_detected = _find_data_usage_in_source(
        simulator,
        candidates,
        analysis,
    )

    detected.update(source_detected)

    for name in candidates:
        if name.lower() in DATA_INPUT_NAMES:
            detected.add(name)

            analysis["evidence"].setdefault(
                name,
                "Classified as fixed because its "
                "name suggests input data or a grid.",
            )

    analysis[
        "fixed_inputs_without_values"
    ].extend(sorted(detected))

    analysis["unclassified"] = [
        name
        for name in analysis["unclassified"]
        if name not in detected
    ]


def _annotation_suggests_data(annotation):
    if annotation is inspect.Parameter.empty:
        return False

    annotation_text = str(annotation).lower()

    data_markers = {
        "ndarray",
        "array",
        "sequence",
        "list",
        "tuple",
        "iterable",
    }

    return any(
        marker in annotation_text
        for marker in data_markers
    )


def _find_data_usage_in_source(
    simulator,
    candidate_names,
    analysis,
):
    tree = _get_source_tree(
        simulator,
        analysis,
        purpose="data-input detection",
    )

    if tree is None:
        return set()

    visitor = DataInputVisitor(candidate_names)
    visitor.visit(tree)

    for name in visitor.detected:
        analysis["evidence"][name] = (
            "Classified as fixed because the "
            "source uses it as array-like or "
            "shape-defining data."
        )

    return visitor.detected


class DataInputVisitor(ast.NodeVisitor):

    def __init__(self, candidate_names):
        self.candidate_names = set(
            candidate_names
        )
        self.detected = set()

    def visit_Attribute(self, node):
        if (
            isinstance(node.value, ast.Name)
            and (
                node.value.id
                in self.candidate_names
            )
            and node.attr in {
                "shape",
                "size",
                "ndim",
            }
        ):
            self.detected.add(
                node.value.id
            )

        self.generic_visit(node)

    def visit_Subscript(self, node):
        if (
            isinstance(node.value, ast.Name)
            and (
                node.value.id
                in self.candidate_names
            )
        ):
            self.detected.add(
                node.value.id
            )

        self.generic_visit(node)

    def visit_Call(self, node):
        self._detect_len_call(node)
        self._detect_array_conversion(node)

        self.generic_visit(node)

    def _detect_len_call(self, node):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and node.args
            and isinstance(
                node.args[0],
                ast.Name,
            )
            and (
                node.args[0].id
                in self.candidate_names
            )
        ):
            self.detected.add(
                node.args[0].id
            )

    def _detect_array_conversion(self, node):
        if not node.args:
            return

        first_argument = node.args[0]

        if not (
            isinstance(first_argument, ast.Name)
            and (
                first_argument.id
                in self.candidate_names
            )
        ):
            return

        function_name = _get_call_name(
            node.func
        )

        if function_name in {
            "array",
            "asarray",
            "zeros_like",
            "ones_like",
        }:
            self.detected.add(
                first_argument.id
            )


def _get_call_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr

    return None


def _classify_remaining_as_direct_inferred(
    analysis,
):
    remaining = list(analysis["unclassified"])

    analysis[
        "direct_inferred_parameters"
    ].extend(remaining)

    for name in remaining:
        analysis["evidence"][name] = (
            "Classified as inferred because it "
            "has no default and was not identified "
            "as RNG or fixed input data."
        )

    analysis["unclassified"] = []


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


def validate_analysis(analysis):
    errors = []

    arguments = set(analysis["arguments"])

    rng_argument = analysis["rng_argument"]
    parameter_container = (
        analysis["parameter_container"]
    )

    fixed_with_values = set(
        analysis["fixed_values"]
    )
    fixed_without_values = set(
        analysis["fixed_inputs_without_values"]
    )
    container_parameters = set(
        analysis["container_parameters"]
    )
    direct_inferred = set(
        analysis["direct_inferred_parameters"]
    )
    combined_inferred = set(
        analysis["inferred_parameters"]
    )
    unclassified = set(
        analysis["unclassified"]
    )

    if rng_argument is not None:
        if rng_argument not in arguments:
            errors.append(
                f"RNG argument {rng_argument!r} "
                "is not in the simulator signature."
            )

    if parameter_container is not None:
        if parameter_container not in arguments:
            errors.append(
                "Parameter container "
                f"{parameter_container!r} is not "
                "in the simulator signature."
            )

        if direct_inferred:
            errors.append(
                "Direct inferred parameters are not "
                "supported when a parameter container "
                "is present."
            )
    elif container_parameters:
        errors.append(
            "Container parameters were found, but no "
            "parameter container is configured."
        )

    top_level_role_sets = {
        "fixed values": fixed_with_values,
        "fixed inputs": fixed_without_values,
        "direct inferred parameters": (
            direct_inferred
        ),
    }

    role_names = list(top_level_role_sets)

    for index, first_name in enumerate(
        role_names
    ):
        for second_name in role_names[
            index + 1:
        ]:
            overlap = (
                top_level_role_sets[first_name]
                & top_level_role_sets[second_name]
            )

            if overlap:
                errors.append(
                    f"Arguments {sorted(overlap)} "
                    f"appear in both {first_name} "
                    f"and {second_name}."
                )

    for category, names in (
        top_level_role_sets.items()
    ):
        invalid = names - arguments

        if invalid:
            errors.append(
                f"{category} contains names not in "
                f"the function signature: "
                f"{sorted(invalid)}"
            )

    special_arguments = {
        name
        for name in {
            rng_argument,
            parameter_container,
        }
        if name is not None
    }

    assigned_top_level = (
        fixed_with_values
        | fixed_without_values
        | direct_inferred
        | special_arguments
    )

    missing_top_level = (
        arguments - assigned_top_level
    )

    if missing_top_level:
        errors.append(
            "Top-level arguments have no role: "
            f"{sorted(missing_top_level)}"
        )

    expected_combined = (
        container_parameters
        | direct_inferred
    )

    if combined_inferred != expected_combined:
        errors.append(
            "The combined inferred parameter list "
            "does not match the container and direct "
            "inferred parameter lists."
        )

    if unclassified:
        errors.append(
            "Unclassified arguments remain: "
            f"{sorted(unclassified)}"
        )

    if errors:
        raise ValueError(
            "Invalid simulator classification:\n- "
            + "\n- ".join(errors)
        )

    return True


def apply_analysis(agent, analysis):
    validate_analysis(analysis)

    if analysis["rng_argument"] is not None:
        agent.set_rng_argument(
            analysis["rng_argument"]
        )

    if (
        analysis["parameter_container"]
        is not None
    ):
        agent.set_parameter_container(
            analysis["parameter_container"]
        )

    agent.set_inferred_parameters(
        *analysis["inferred_parameters"]
    )

    if analysis["fixed_values"]:
        agent.set_fixed_values(
            **analysis["fixed_values"]
        )

    return agent