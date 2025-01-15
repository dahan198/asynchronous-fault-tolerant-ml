import torch
from .aggregator import Aggregator


class WeightedCWMed(Aggregator):
    """
    Implements a weighted coordinate-wise median aggregation for federated learning.

    This method computes the weighted median of gradients across clients, ensuring robustness
    against outliers by considering the cumulative sum of normalized weights.
    """

    def __init__(self, **kwargs):
        """
        Initializes the WeightedCWMed instance. Additional parameters can be passed through kwargs.
        """
        pass

    def __call__(self, gradients, weights=None, *args, **kwargs):
        """
        Aggregates gradients using the weighted coordinate-wise median.

        Args:
            gradients (torch.Tensor): A tensor containing gradients from different clients.
            weights (torch.Tensor): A tensor containing the weights associated with each client's gradients.

        Returns:
            torch.Tensor: A tensor representing the aggregated gradients using the weighted median.
        """
        # Normalize the weights so that they sum to 1
        weights = weights / weights.sum()

        # Sort gradients along the first dimension (clients) for each feature
        sorted_indices = torch.argsort(gradients, dim=0)
        sorted_gradients = torch.gather(gradients, 0, sorted_indices)

        # Sort weights according to the sorted gradient indices
        sorted_weights = weights[sorted_indices].squeeze(2)

        # Compute the cumulative sum of weights for each dimension
        cum_weights = torch.cumsum(sorted_weights, dim=0)

        # Find the index where the cumulative weight exceeds 0.5 for each feature
        median_idx = torch.maximum(
            torch.argmin((cum_weights <= 0.5).int(), dim=0),
            torch.zeros_like(gradients[0, :])
        ).int()

        # Select the median gradients based on the computed median indices
        weighted_median = sorted_gradients[
            median_idx, torch.arange(gradients.size(1), device=gradients.device)
        ]

        return weighted_median
