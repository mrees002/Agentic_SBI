import numpy as np


def simulate_normal(mean, sd, rng, n=100):
    return rng.normal(
        loc=mean,
        scale=sd,
        size=n,
    )


def simulate_uniform(lower, upper, n=100):
    if lower >= upper:
        raise ValueError(
            "lower must be less than upper"
        )

    return np.random.uniform(
        low=lower,
        high=upper,
        size=n,
    )


def simulate_lognormal(mu, sigma, n=200):
    return np.random.lognormal(
        mean=mu,
        sigma=sigma,
        size=n,
    )


def simulate_poisson(rate, n=100):
    return np.random.poisson(
        lam=rate,
        size=n,
    )


def simulate_quadratic(
    coefficient,
    x,
    noise_sd,
):
    return (
        coefficient * x**2
        + np.random.normal(
            0,
            noise_sd,
            size=len(x),
        )
    )


def simulate_decay(
    rate,
    time,
    rng,
    noise_sd=0.1,
):
    time = np.asarray(
        time,
        dtype=float,
    )

    if time.ndim != 1:
        raise ValueError(
            "time must be one-dimensional"
        )

    return (
        np.exp(-rate * time)
        + rng.normal(
            0,
            noise_sd,
            size=time.shape,
        )
    )

def simulate_container(theta, rng, n=100):
    return rng.normal(
        theta["mu"],
        theta["sigma"],
        size=n,
    )

def simulate_stochastic_logistic_growth(
    growth_rate,
    carrying_capacity,
    rng,
    initial_population=20.0,
    n_steps=100,
    time_step=0.1,
    noise_scale=0.5,
):
    if growth_rate <= 0:
        raise ValueError(
            "growth_rate must be greater than zero."
        )

    if carrying_capacity <= 0:
        raise ValueError(
            "carrying_capacity must be greater than zero."
        )

    if initial_population <= 0:
        raise ValueError(
            "initial_population must be greater than zero."
        )

    population = np.empty(
        n_steps,
        dtype=float,
    )

    population[0] = initial_population

    for index in range(1, n_steps):
        current = population[index - 1]

        deterministic_change = (
            growth_rate
            * current
            * (1.0 - current / carrying_capacity)
            * time_step
        )

        stochastic_change = (
            noise_scale
            * np.sqrt(max(current, 0.0))
            * np.sqrt(time_step)
            * rng.normal()
        )

        population[index] = max(
            current
            + deterministic_change
            + stochastic_change,
            0.0,
        )

    return population