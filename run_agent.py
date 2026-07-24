from agent.function_analyzer import (
    analyze_agent,
    apply_analysis,
    validate_analysis,
)

from agent.prompts import (
    ask_config_path,
    ask_simulator_function_name,
    ask_simulator_path,
    ask_use_config,
    collect_missing_inputs,
    review_analysis,
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
    config = load_config_file(config_path)

    simulator_settings = config["simulator"]

    simulator_path = simulator_settings.get(
        "path"
    )
    simulator_name = simulator_settings.get(
        "name"
    )

    if not simulator_path:
        raise ValueError(
            "The config does not contain "
            "a simulator path."
        )

    if not simulator_name:
        raise ValueError(
            "The config does not contain "
            "a simulator function name."
        )

    simulator = load_simulator(
        simulator_path,
        simulator_name,
    )

    agent = SimulatorAgent(simulator)

    agent.set_simulator_path(
        simulator_path
    )

    agent.configure_from_file(
        config_path
    )

    synthetic_config_path = (
        Path(config_path).parent
        / "synthetic_config.json"
    )

    synthetic_config = (
        load_synthetic_config(
            synthetic_config_path
        )
    )

    if synthetic_config is not None:
        agent.set_synthetic_metadata(
            true_parameter_values=(
                synthetic_config[
                    "true_parameter_values"
                ]
            ),
            generation_seed=(
                synthetic_config[
                    "generation_seed"
                ]
            ),
        )

    return agent

def create_agent_from_config_interactively():
    while True:
        config_path = ask_config_path()

        try:
            agent = create_agent_from_config(
                config_path
            )
        except (
            FileNotFoundError,
            KeyError,
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
        ) as error:
            print(
                "\nCould not load configuration:"
            )
            print(error)
            print(
                "Please select a config "
                "file again.\n"
            )
            continue

        return agent, config_path

def create_agent_interactively():
    while True:
        simulator_path = ask_simulator_path()

        while True:
            simulator_name = input(
                "Simulator function name "
                "or 'back' to choose another file: "
            ).strip()

            if simulator_name.lower() == "back":
                break

            if not simulator_name:
                print(
                    "A simulator function name "
                    "is required."
                )
                continue

            if not simulator_name.isidentifier():
                print(
                    "Function name must be a valid "
                    "Python identifier."
                )
                continue

            try:
                simulator = load_simulator(
                    simulator_path,
                    simulator_name,
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
            ) as error:
                print(
                    "\nCould not load simulator:"
                )
                print(error)
                print()
                continue

            agent = SimulatorAgent(simulator)
            agent.set_simulator_path(
                simulator_path
            )

            analysis = analyze_agent(agent)

            confirmed_analysis = (
                review_analysis(analysis)
            )

            validate_analysis(
                confirmed_analysis
            )

            apply_analysis(
                agent,
                confirmed_analysis,
            )

            collect_missing_inputs(agent)

            return agent

def main():
    use_config = ask_use_config()

    if use_config:
        agent, source_config_path = (
            create_agent_from_config_interactively()
            )

        print(
            "\nConfiguration loaded."
        )

    else:
        agent = create_agent_interactively()

        print(
            "\nInteractive configuration "
            "completed."
        )

    remaining = agent.get_missing_fields()

    if remaining:
        raise ValueError(
            "Agent configuration is incomplete: "
            f"{remaining}"
        )

    run_paths = create_run_directory(
        simulator_name=(
            agent.simulator.__name__
        ),
    )

    save_run_data(
        agent=agent,
        run_directory=(
            run_paths["run_directory"]
        ),
    )

    agent.create_config(
        run_paths["config_path"]
    )

    if agent.true_parameter_values is not None:
        create_synthetic_config(
            output_path=(
                run_paths[
                    "synthetic_config_path"
                ]
            ),
            true_parameter_values=(
                agent.true_parameter_values
            ),
            generation_seed=(
                agent.synthetic_generation_seed
            ),
        )

    print(
        "Run directory:",
        run_paths["run_directory"],
    )

    print(
        "Config saved to:",
        run_paths["config_path"],
    )

    agent.build_wrapper()

    validation_report = agent.test_abc()

    print("\nValidation report:")
    for key, val in validation_report.items():
        print(f"{key}: {val}")

    accepted_parameters, accepted_distances = (
        agent.run_abc()
    )

    accepted_count = len(
        accepted_parameters
    )

    print("\nABC inference completed.")

    print(
        "Accepted samples:",
        accepted_count,
    )

    print(
        "Acceptance rate:",
        accepted_count
        / agent.n_simulations,
    )

    results, results_path = save_results(
        agent=agent,
        config_path=run_paths["config_path"],
        output_path=run_paths["results_path"],
    )

    agent.plot_posterior_hist(
        run_paths["posterior_plot_path"]
    )

    agent.plot_distance_hist(
        run_paths["distance_plot_path"]
    )

    print(
        "Results saved to:",
        results_path,
    )

    print(
        "Plots saved in:",
        run_paths["run_directory"],
    )

if __name__ == "__main__":
    main()