import torch
from .aggregator import Aggregator


class CWMed(Aggregator):
    """
    Implements the Coordinate-Wise Median (CWMed) aggregation method for distributed learning.
    This method computes the median of gradients along each coordinate, providing robustness
    against Byzantine workers.
    """

    def __init__(self, **kwargs):
        """
        Initializes the CWMed aggregator. No specific parameters are required for this method,
        but additional keyword arguments are accepted for compatibility with other aggregators.

        Args:
            **kwargs: Arbitrary keyword arguments (not used in CWMed).
        """
        pass

    def __call__(self, gradients, *args, **kwargs):
        """
        Applies the coordinate-wise median aggregation on the provided gradients.

        Args:
            gradients (torch.Tensor): A tensor of shape (num_workers, num_parameters) containing
                                      the gradients from each worker.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The aggregated gradient obtained by computing the median along each coordinate.
                - None: A placeholder for compatibility with other aggregators.
        """
        aggregated_gradient = torch.median(gradients, dim=0)[0]
        return aggregated_gradient
