import numpy as np

def simulate_normal(
    mean,
    standard_deviation,
    sample_size=100,
    rng=None,
):
    return rng.normal(
        mean,
        standard_deviation,
        size=int(sample_size),
    )

def simulate_regression(
    slope,
    intercept,
    x,
    noise_sd=0.2,
    rng=None,
):
    return (
        intercept
        + slope * x
        + rng.normal(
            0,
            noise_sd,
            size=len(x),
        )
    )

def simulate_decay(
    decay_rate,
    observation_points,
    random_source,
    initial_value,
    noise_sd
):
    mean_curve = initial_value * np.exp(-decay_rate * observation_points)

    noise = random_source.normal(
        loc=0.0,
        scale=noise_sd,
        size=observation_points.shape,
    )

    return mean_curve + noise