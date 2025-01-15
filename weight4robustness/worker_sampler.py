import random
import numpy as np


def sample_worker_by_id(n):
    """
    Samples a worker based on ID, where the probability of selecting a worker
    is proportional to its ID.

    Args:
        n (int): Total number of workers.

    Returns:
        int: The index of the selected worker.
    """
    # Calculate the total sum of all IDs (1 to n)
    total_sum = (n * (n + 1)) // 2

    # Generate probabilities proportional to ID numbers
    probabilities = [i / total_sum for i in range(1, n + 1)]

    # Select a worker based on these probabilities
    chosen_worker = random.choices(range(n), weights=probabilities, k=1)

    return chosen_worker[0]


def uniform_sample_worker(n):
    """
    Samples a worker uniformly, where each worker has an equal probability of being selected.

    Args:
        n (int): Total number of workers.

    Returns:
        int: The index of the selected worker.
    """
    # Uniform probabilities for all workers
    probabilities = [1 / n] * n

    # Select a worker with equal probability
    chosen_worker = random.choices(range(n), weights=probabilities, k=1)

    return chosen_worker[0]


def sample_worker_by_id_square(n):
    """
    Samples a worker based on the square of its ID, where the probability
    of selecting a worker is proportional to the square of its ID.

    Args:
        n (int): Total number of workers.

    Returns:
        int: The index of the selected worker.
    """
    # Generate weights proportional to the square of IDs
    weights = [i ** 2 for i in range(1, n + 1)]

    # Compute the total sum of weights
    total_sum = np.sum(weights)

    # Generate probabilities proportional to squared ID weights
    probabilities = [i / total_sum for i in weights]

    # Select a worker based on squared ID probabilities
    chosen_worker = random.choices(range(n), weights=probabilities, k=1)

    return chosen_worker[0]


# Dictionary mapping sampler types to their corresponding functions
SAMPLER = {
    'id': sample_worker_by_id,
    'id2': sample_worker_by_id_square,
    'uniform': uniform_sample_worker,
}
