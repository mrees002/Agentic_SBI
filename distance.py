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

    return np.linalg.norm(np.array(summary1) - np.array(summary2))