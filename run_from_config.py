from agent.simulator_agent import SimulatorAgent
from simulators import simulate_exponential_decay


def main():
    agent = SimulatorAgent(
        simulate_exponential_decay
    )

    agent.configure_from_file(
        "generated_config.json"
    )

    print("Configuration loaded.")
    print(
        "Inferred parameters:",
        agent.inferred_parameters,
    )
    print(
        "Fixed values:",
        list(agent.fixed_values),
    )
    print(
        "Remaining fields:",
        agent.get_missing_fields(),
    )

    agent.build_wrapper()

    validation_report = agent.test_abc()

    print("\nValidation report:")
    print(validation_report)

    accepted_parameters, accepted_distances = (
        agent.run_abc()
    )

    print("\nABC inference completed.")
    print(
        "Accepted samples:",
        len(accepted_parameters),
    )
    print(
        "Acceptance rate:",
        len(accepted_parameters)
        / agent.n_simulations,
    )

    agent.plot_posterior_hist(
        "posterior_from_config.png"
    )


if __name__ == "__main__":
    main()