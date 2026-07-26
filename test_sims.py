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