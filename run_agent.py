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

from agent.config import(
    load_config_file
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

    return agent

def create_agent_interactively():
    simulator_path = ask_simulator_path()

    simulator_name = (
        ask_simulator_function_name()
    )

    simulator = load_simulator(
        simulator_path,
        simulator_name,
    )

    agent = SimulatorAgent(simulator)

    agent.set_simulator_path(
        simulator_path
    )

    analysis = analyze_agent(agent)

    confirmed_analysis = review_analysis(
        analysis
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
        source_config_path = (
            ask_config_path()
        )

        agent = create_agent_from_config(
            source_config_path
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
            run_paths["directory"]
        ),
    )

    agent.create_config(
        run_paths["config_path"]
    )

    print(
        "Run directory:",
        run_paths["directory"],
    )

    print(
        "Config saved to:",
        run_paths["config_path"],
    )

    agent.build_wrapper()

    validation_report = agent.test_abc()

    print("\nValidation report:")
    print(validation_report)

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
        run_paths["posterior_path"]
    )

    print(
        "Results saved to:",
        results_path,
    )

    print(
        "Posterior plot saved in:",
        run_paths["directory"],
    )

if __name__ == "__main__":
    main()