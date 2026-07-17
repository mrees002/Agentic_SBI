import numpy as np

def euclidean_distance(summary1, summary2):

    """
    Computes the Euclidean distance between two summary statistics.

    Parameters:
    summary1: Summary statistics of the observed data
    summary2: Summary statistics of the simulated data

    Returns:
    float: The Euclidean distance between the two summary statistics.
    """

    summary1 = np.asarray(summary1, dtype=float)
    summary2 = np.asarray(summary2, dtype=float)

    if summary1.shape != summary2.shape:
        raise ValueError("Summaries must have the same shape.")

    if not np.all(np.isfinite(summary1)):
        raise ValueError("First summary contains non-finite values.")

    if not np.all(np.isfinite(summary2)):
        raise ValueError("Second summary contains non-finite values.")

    return float(np.linalg.norm(summary1 - summary2))