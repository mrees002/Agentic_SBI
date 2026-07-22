from agent.function_analyzer import (
    analyze_agent,
    apply_analysis,
    validate_analysis,
)

from agent.prompts import (
    ask_simulator_function_name,
    ask_simulator_path,
    collect_missing_inputs,
    display_analysis,
    review_analysis,
)

from agent.simulator_agent import (
    SimulatorAgent,
)

from agent.simulator_loader import (
    load_simulator,
)



def main():
    simulator_path = ask_simulator_path()

    function_name = (
        ask_simulator_function_name()
    )

    simulator = load_simulator(
        simulator_path,
        function_name,
    )

    print(
        f"\nLoaded simulator "
        f"{simulator.__name__!r} "
        f"from {simulator_path!r}."
    )

    agent = SimulatorAgent(simulator)

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

    agent.create_config("generated_config.json")
    agent.build_wrapper()

    validation_report = agent.test_abc()

    print("\nValidation report:")
    print(validation_report)

    accepted_parameters, accepted_distances = (
        agent.run_abc()
    )

    print("\nABC inference completed.")
    print(
        f"Accepted samples: "
        f"{len(accepted_parameters)}"
    )
    print(
        f"Acceptance rate: "
        f"{len(accepted_parameters) / agent.n_simulations:.4f}"
    )

    agent.plot_posterior_hist(
        "posterior_histograms.png"
    )


if __name__ == "__main__":
    main()