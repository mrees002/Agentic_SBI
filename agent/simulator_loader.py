import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable


def load_simulator(
    file_path,
    function_name,
):
    """
    Load a simulator function from a Python file.

    Parameters
    ----------
    file_path:
        Path to a Python source file.

    function_name:
        Name of the simulator function defined
        inside that file.
    """
    path = Path(file_path).expanduser()

    _validate_simulator_path(path)

    module = _load_module_from_path(path)

    simulator = _get_simulator_function(
        module,
        function_name,
        path,
    )

    return simulator


def _validate_simulator_path(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Simulator file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Simulator path is not a file: {path}"
        )

    if path.suffix.lower() != ".py":
        raise ValueError(
            "Simulator file must be a Python "
            f".py file: {path}"
        )


def _load_module_from_path(path):
    module_name = (
        f"_agentic_sbi_simulator_"
        f"{abs(hash(path.resolve()))}"
    )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            "Could not create an import "
            f"specification for {path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    try:
        specification.loader.exec_module(
            module
        )
    except Exception as error:
        raise ImportError(
            "Simulator file could not be "
            f"executed: {path}\n"
            f"Original error: {error}"
        ) from error

    return module


def _get_simulator_function(
    module: ModuleType,
    function_name: str,
    path: Path,
) -> Callable:
    if not isinstance(function_name, str):
        raise TypeError(
            "function_name must be a string."
        )

    function_name = function_name.strip()

    if not function_name:
        raise ValueError(
            "A simulator function name is required."
        )

    if not hasattr(module, function_name):
        available_functions = (
            _find_public_functions(module)
        )

        raise AttributeError(
            f"Function {function_name!r} was not "
            f"found in {path}. "
            "Available functions: "
            f"{available_functions}"
        )

    simulator = getattr(
        module,
        function_name,
    )

    if not callable(simulator):
        raise TypeError(
            f"{function_name!r} exists in {path}, "
            "but it is not callable."
        )

    return simulator


def _find_public_functions(module):
    names = []

    for name, value in vars(module).items():
        if name.startswith("_"):
            continue

        if callable(value):
            names.append(name)

    return sorted(names)