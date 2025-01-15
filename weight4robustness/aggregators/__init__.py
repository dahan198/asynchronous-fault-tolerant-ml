from .cwmed import CWMed
from .rfa import RobustFederatedAveraging
from .weighted_rfa import WeightedRobustFederatedAveraging
from .weighted_cwmed import WeightedCWMed


# Registry mapping aggregator names to their respective classes
AGGREGATOR_REGISTRY = {
    'cwmed': CWMed,
    'rfa': RobustFederatedAveraging,
    'weighted_rfa': WeightedRobustFederatedAveraging,
    'weighted_cwmed': WeightedCWMed,
}


def get_aggregator(aggregation, num_workers, num_byzantine, lambda_byz=None):
    """
    Retrieves and initializes an aggregator instance based on the specified parameters.

    Args:
        aggregation (str): The name of the aggregation method to be used.
        num_workers (int): The total number of workers participating in the aggregation.
        num_byzantine (int): The number of Byzantine (malicious) workers.
        lambda_byz (float, optional): The fraction of Byzantine iterations (if applicable).

    Returns:
        object: An instance of the selected aggregator class, initialized with the provided parameters.

    Raises:
        AssertionError: If the specified aggregation method is not found in `AGGREGATOR_REGISTRY`.
    """
    assert aggregation in AGGREGATOR_REGISTRY, f"{aggregation} is unknown or unsupported."

    # Common parameters for all aggregators
    common_params = {
        "num_workers": num_workers,
        "num_byzantine": num_byzantine,
        "lambda_byz": lambda_byz,
    }

    # Instantiate and return the appropriate aggregator
    return AGGREGATOR_REGISTRY[aggregation](**common_params)
