from agent.function_analyzer import (
    analyze_agent,
    apply_analysis,
    validate_analysis,
)
from agent.prompts import (
    collect_missing_inputs,
    display_analysis,
    review_analysis,
)
from agent.simulator_agent import (
    SimulatorAgent,
)

from simulators import (
    simulate_exponential_decay,
)


def main():
    agent = SimulatorAgent(
        simulate_exponential_decay
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