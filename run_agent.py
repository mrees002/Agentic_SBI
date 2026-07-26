from agent.function_analyzer import (
    analyze_agent,
    apply_analysis,
    validate_analysis,
)

from agent.prompts import (
    ask_config_path,
    ask_simulator_path,
    ask_simulator_function_name,
    ask_use_config,
    collect_missing_inputs,
    review_analysis,
    revise_fixed_value,
    revise_prior_bounds,
    ask_validation_adjustment
)

from agent.config import (
    create_synthetic_config,
    load_config_file,
    load_synthetic_config,
)

from agent.simulator_agent import (
    SimulatorAgent,
)

from agent.simulator_loader import (
    load_simulator,
)

from agent.results import(
    save_results
)

from agent.run_directory import (
    create_run_directory,
    save_run_data,
)

from pathlib import Path

def create_agent_from_config(config_path):

    # load file from config
    config = load_config_file(config_path)

    # get sim settings and sim names
    simulator_settings = config["simulator"]
    simulator_path = simulator_settings.get("path")
    simulator_name = simulator_settings.get("name")

    if not simulator_path:
        raise ValueError("The config does not contain a simulator path.")

    if not simulator_name:
        raise ValueError("The config does not contain a simulator function name.")

    simulator = load_simulator(simulator_path, simulator_name)
    agent = SimulatorAgent(simulator)
    agent.set_simulator_path(simulator_path)
    agent.configure_from_file(config_path)

    # set synthetic data settings if needed
    synthetic_config_path = (Path(config_path).parent / "synthetic_config.json")
    synthetic_config = (load_synthetic_config(synthetic_config_path))

    if synthetic_config is not None:
        agent.set_synthetic_metadata(
            true_parameter_values=(synthetic_config["true_parameter_values"]),
            generation_seed=(synthetic_config["generation_seed"]),
        )

    return agent

def create_agent_from_config_interactively():
    # try to load agent from config
    while True:
        config_path = ask_config_path()

        # try until no errors
        try:
            agent = create_agent_from_config(config_path)
        except (FileNotFoundError, KeyError, ImportError, AttributeError, TypeError, ValueError) as error:
            print("\nCould not load configuration:")
            print(error)
            print("Please select a config file again.\n")
            continue

        return agent, config_path

def create_agent_interactively():
    # get simulator path
    while True:
        simulator_path = ask_simulator_path()

        # get simulator name
        while True:
            simulator_name = ask_simulator_function_name(allow_back = True)
            if simulator_name is None:
                break
            
            try:
                simulator = load_simulator(simulator_path, simulator_name)
            except (AttributeError, TypeError, ValueError) as error:
                print("\nCould not load simulator:")
                print(error)
                print()
                continue

            # set agent corresponding to simulator selected
            agent = SimulatorAgent(simulator)
            agent.set_simulator_path(simulator_path)

            # analyze agent and ensure values are correct
            analysis = analyze_agent(agent)
            confirmed_analysis = (review_analysis(analysis))
            validate_analysis(confirmed_analysis)

            # apply changes to agent if needed
            apply_analysis(agent, confirmed_analysis)

            if agent.rng_argument is None:
                print(
                    "\nWarning: this simulator does not "
                    "accept an RNG argument. Repeated runs "
                    "may not be reproducible.\n"
                )

            # collect missing inputs if any
            configured_agent = collect_missing_inputs(agent)

            if configured_agent is None:
                print("\nInteractive setup was cancelled. Please start the setup again.\n")
                break

            return configured_agent

def main():
    # check if user wants to load config
    use_config = ask_use_config()

    # create agent corresponding to config load
    if use_config:
        agent, source_config_path = (create_agent_from_config_interactively())
        print("\nConfiguration loaded.")

    else:
        agent = create_agent_interactively()
        print("\nInteractive configuration completed.")

    # check no fields missing
    remaining = agent.get_missing_fields()
    if remaining:
        raise ValueError(f"Agent configuration is incomplete: {remaining}")

    agent.build_wrapper()

    # validate agent is built correctly
    while True:
        validation_report = agent.test_abc()

        if validation_report["success"]:
            print("\nValidation report:")

            for key, value in (validation_report.items()):
                print(f"{key}: {value}")

            break

        adjustment = ask_validation_adjustment(agent, validation_report)

        if adjustment is None:
            print("\nValidation cancelled. No run directory or config was created.")
            return

        if adjustment == "prior_bounds":
            revise_prior_bounds(agent)

        elif adjustment == "fixed_values":
            revise_fixed_value(agent)

    # create directory and save config
    run_paths = create_run_directory(simulator_name=(agent.simulator.__name__))
    save_run_data(agent=agent, run_directory=(run_paths["run_directory"]))
    agent.create_config(run_paths["config_path"])

    # create a config file for synthetic data if needed
    if agent.true_parameter_values is not None:
        create_synthetic_config(
            output_path=(run_paths["synthetic_config_path"]),
            true_parameter_values=(agent.true_parameter_values),
            generation_seed=(agent.synthetic_generation_seed),
        )

    # print directory
    print("Run directory:", run_paths["run_directory"])
    print("Config saved to:", run_paths["config_path"])

    # notify user if error occurs with abc sims and break if so
    try:
        accepted_parameters, accepted_distances = agent.run_abc()
    except ValueError as error:
        print(f"\nABC inference unsuccessful: {error}")
        return

    # print simple stats
    accepted_count = len(accepted_parameters)
    print("\nABC inference completed.")
    print("Accepted samples:", accepted_count)
    print("Acceptance rate:", accepted_count / agent.n_simulations)

    # save results
    results, results_path = save_results(
        agent=agent,
        config_path=run_paths["config_path"],
        output_path=run_paths["results_path"],
    )

    # create plots
    agent.plot_posterior_hist(run_paths["posterior_plot_path"])
    agent.plot_parameter_pairs(run_paths["parameter_pairs_plot_path"])

    print("Results saved to:", results_path)
    print("Plots saved in:", run_paths["run_directory"])

if __name__ == "__main__":
    main()