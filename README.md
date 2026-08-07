# Agentic SBI

Agentic SBI is a small Python framework for building a reproducible simulation-based inference workflow around an existing simulator. It inspects a simulator's function signature, proposes how its arguments should be classified, collects the missing configuration, validates the resulting setup, runs rejection Approximate Bayesian Computation (ABC), and saves the results in a structured run directory.

The current implementation is intended as a course and research project rather than a general-purpose SBI library.

## Overview

The workflow is:

```text
Python simulator
    |
    v
Automatic signature analysis
    |
    v
User review and missing inputs
    |
    v
Observed or synthetic data
    |
    v
Pre-run validation
    |
    v
Rejection ABC
    |
    v
Configuration, results, and plots
```

The project currently supports:

- direct simulator parameters;
- automatic detection of common random-number-generator argument names;
- user confirmation of an RNG argument when one is not detected automatically;
- interactive correction of proposed argument classifications;
- scalar and NumPy-array fixed inputs;
- synthetic or externally supplied observed data;
- independent uniform priors;
- rejection ABC;
- validation before committing to the full simulation budget;
- guided revision of prior bounds, fixed values, and synthetic true values;
- reproducible run directories containing data, configuration, results, and plots.

## Quick start

Create a simulator file such as `example_simulator.py`:

```python
import numpy as np

def simulate_normal(mean, standard_deviation, rng, n_observations=100):
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be greater than zero.")

    return rng.normal(mean, standard_deviation, size=n_observations)
```

Run the agent:

```bash
python run_agent.py
```

A typical interactive session would use inputs similar to:

```text
Use an existing config file? [y/n]: n
Path to simulator Python file: example_simulator.py
Simulator function name or 'back' to choose another file: simulate_normal
Is this classification correct? [y/n]: y

Lower prior bound for mean: -5
Upper prior bound for mean: 5

Lower prior bound for standard_deviation: 0.1
Upper prior bound for standard_deviation: 2

ABC epsilon: 1
Number of simulations: 5000
Random seed [123]:

Generate synthetic observed data? [y/n]: y
True value for mean: 0
True value for standard_deviation: 1
```

The agent will generate synthetic observed data, validate the simulator and ABC configuration, run rejection ABC, and save the run outputs under `runs/`.

## Simulator requirements

A simulator must be a Python callable loaded from a `.py` file.

A supported simulator uses direct named arguments, for example:

```python
def simulator(inferred_parameter, fixed_input, rng):
    ...
```

Common RNG argument names are detected automatically: `rng`, `random_state`, and `generator`. If no recognized name is found, the agent asks the user whether one of the simulator arguments should be treated as the RNG argument.

A supported simulator should:

- return a nonempty numeric value or NumPy-compatible array;
- return the same output shape for simulated and observed data;
- avoid NaN and infinite outputs;
- use explicit named arguments;
- avoid `*args` and `**kwargs`;
- accept an RNG argument when reproducibility is required.

Simulators without an RNG argument can still run, but repeated executions may not be reproducible.

The current wrapper expects inferred parameters to appear directly in the simulator signature. Parameter-container interfaces such as `theta["parameter"]` are not currently supported by the agent.

## Automatic argument analysis

The analyzer uses the simulator function signature and simple argument-name rules to propose roles for the simulator inputs.

The current logic is:

- recognized RNG names are classified as the RNG argument;
- arguments with Python default values are classified as fixed values;
- recognized data-input names such as `x`, `time`, `grid`, `inputs`, `features`, and `design_matrix` are suggested as fixed inputs;
- remaining arguments are classified as inferred parameters.

The analyzer does not inspect the simulator source code to discover how arguments are used.

The proposed classification is shown before configuration continues. The user can move a top-level argument between an inferred parameter and a fixed input requiring a value.

## Configuration workflows

### Interactive configuration

The interactive workflow collects any missing information, including:

- inferred-parameter names;
- fixed input values;
- prior bounds;
- ABC tolerance;
- number of simulations;
- random seed;
- observed data.

Fixed inputs can be supplied as:

1. a numeric scalar;
2. a `.npy` array;
3. an evenly spaced numeric sequence.

### Existing configuration

A previous `config.json` can be loaded instead of rebuilding the setup interactively.

Relative paths in the configuration are resolved from the directory containing the configuration file. This allows a saved run directory to remain portable as long as its referenced files stay together.

When a previous configuration is loaded, the simulator path and function name are read from the config, the agent is rebuilt, and a new run directory is created for the new inference run.

## Observed data

The agent supports two observed-data workflows.

### Load existing data

Observed data can be loaded from a `.npy` file. The prompt checks that the file exists, can be loaded without pickled objects, is not empty, and contains only finite values.

### Generate synthetic data

The agent can generate observed data by running the simulator with user-provided true parameter values.

The default seed behavior is:

```text
random_seed      -> validation and ABC
random_seed + 1  -> synthetic observed-data generation
```

This prevents the synthetic dataset and the inference run from starting from the exact same pseudorandom sequence.

## Validation and guided revision

Before running the full ABC budget, the agent performs a small validation run using several prior draws. The current default is 10 validation checks.

Validation checks include:

- observed data can be converted to numeric form;
- observed data are nonempty and finite;
- the prior can be sampled;
- the simulator executes successfully;
- simulator outputs are numeric, finite, and nonempty;
- simulated and observed output shapes agree;
- the summary function executes successfully;
- simulated and observed summary shapes agree;
- the distance function returns a finite numeric value.

When a recoverable failure occurs, the agent may offer to revise prior bounds, a fixed value, or a synthetic true parameter value before validation is attempted again.

## Inference

The current inference method is rejection ABC.

For each simulation:

1. sample a parameter vector from the prior;
2. run the simulator;
3. summarize simulated and observed data;
4. calculate the distance between the summaries;
5. accept the parameter vector when the distance is below epsilon.

The current defaults are:

```text
Prior: independent uniform bounds
Summary: mean and standard deviation
Distance: Euclidean distance
Algorithm: rejection ABC
```

A smaller epsilon requires closer agreement but usually lowers the acceptance rate. Increasing the simulation budget reduces Monte Carlo variability and produces a more stable approximation, but it does not remove bias caused by a large epsilon or recover information missing from the selected summary statistics.

### Summary-statistic limitation

The current implementation always uses the mean and standard deviation. These summaries are convenient, but they may not identify every parameter in a simulator.

For example, in linear regression the mean and standard deviation of the simulated outputs do not preserve the full relationship between `x` and `y`. Different slope and intercept combinations can therefore produce similar summaries, especially when the input values are far from zero.

A successful ABC run means that accepted simulations matched the selected summaries within epsilon. It does not guarantee that each model parameter is individually identifiable.

## Output directory

Each successful run creates a numbered directory:

```text
runs/
└── simulator_name_YYYYMMDD_001/
    ├── config.json
    ├── observed_data.npy
    ├── results.json
    ├── posterior.png
    ├── parameter_pairs.png
    └── synthetic_config.json
```

Some files are conditional:

- `synthetic_config.json` is created when observed data were generated synthetically;
- `parameter_pairs.png` is created only when at least two parameters are inferred;
- array-valued fixed inputs are copied into the run directory as additional `.npy` files such as `x.npy`.

## Results

`results.json` contains:

- simulator and configuration references;
- number of accepted samples;
- acceptance rate;
- posterior summaries for each inferred parameter;
- accepted-distance summaries.

Parameter summaries include mean, standard deviation, median, minimum, maximum, and an approximate 95% credible interval.

## Configuration files

A run configuration records:

- simulator name and path;
- RNG argument;
- inferred parameters;
- prior bounds;
- literal and file-backed fixed values;
- observed-data path;
- epsilon;
- simulation count;
- random seed;
- summary-function name;
- distance-function name.

The current code records the summary and distance names for reproducibility and documentation. It does not yet dynamically select alternate functions from those names when loading a configuration.

Synthetic observed-data settings are stored separately in `synthetic_config.json`, including the true parameter values and generation seed.

## Repository structure

```text
run_agent.py
    Main interactive orchestration.

agent/simulator_loader.py
    Loads a simulator function from a Python file.

agent/function_analyzer.py
    Inspects the simulator signature and proposes argument roles.

agent/prompts.py
    Handles interactive input, classification review, and revision.

agent/simulator_agent.py
    Stores configuration, builds the simulator wrapper, validates the setup, and runs ABC.

agent/config.py
    Creates, validates, loads, and normalizes configuration files.

agent/run_directory.py
    Creates run directories and saves observed and fixed data.

agent/results.py
    Builds and saves posterior and distance summaries.

agent/serialization.py
    Converts NumPy and Path values into JSON-safe forms.

prior.py
    Implements the independent uniform prior.

inference.py
    Implements rejection ABC.

summaries.py
    Implements the default mean-and-standard-deviation summary.

distance.py
    Implements Euclidean distance.

plots.py
    Creates posterior and parameter-pair visualizations.

posterior.py
    Calculates and samples the exact Gaussian regression posterior used as a benchmark.

simulators.py
    Contains simple example simulators.
```

## Example model types

The repository currently includes simple examples such as:

- linear regression with Gaussian noise;
- a normal-distribution simulator;
- exponential decay with Gaussian noise.

Simple models with one or two identifiable parameters are best for initial testing because the current summary and distance functions are intentionally limited.

## Reproducibility

When a simulator accepts the supplied NumPy RNG, repeated runs with the same configuration and seed should reproduce the same pseudorandom sequence.

The agent uses `np.random.default_rng(random_seed)` for validation and ABC. Synthetic observed data use `np.random.default_rng(random_seed + 1)`.

Reproducibility cannot be guaranteed for simulators that create or use an unrelated random generator internally.

## Limitations

The current implementation is deliberately narrow.

- Only rejection ABC is implemented.
- Priors are independent uniform distributions.
- Summary statistics are fixed to mean and standard deviation.
- Euclidean distance is fixed as the comparison metric.
- Simulator analysis is based on the function signature and a limited set of recognized argument names.
- Parameter-container interfaces are not supported.
- There is no automated parameter-identifiability analysis.
- Structured multivariate or time-series outputs do not have specialized summaries.
- ABC-SMC and other sequential methods are not implemented.
- There is no automatic epsilon-selection procedure.
- The user cannot currently select alternative result plots through the agent.

## Future work

Possible extensions include:

- selectable built-in summary functions;
- user-supplied summary and distance functions;
- summary scaling or normalized Euclidean distance;
- additional prior families;
- automatic or guided epsilon selection;
- quick comparison of multiple epsilon values and simulation budgets;
- ABC-SMC;
- multidimensional and time-series summaries;
- broader simulator-interface support;
- identifiability diagnostics;
- more flexible result plotting.

## Project status

The project is under active development as an academic implementation of an agent-guided SBI workflow. It is suitable for controlled experiments and demonstrations, but results should be interpreted with attention to the chosen prior, tolerance, simulation budget, summary statistics, and parameter identifiability.
