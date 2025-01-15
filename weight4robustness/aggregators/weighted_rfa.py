import torch
from .rfa import RobustFederatedAveraging


class WeightedRobustFederatedAveraging(RobustFederatedAveraging):
    """
    Extends the RobustFederatedAveraging class to incorporate weighted aggregation
    in the smoothed Weiszfeld method.

    This implementation supports both uniform weighting (if no weights are provided)
    and custom weighting, where weights are normalized before aggregation.
    """

    def __init__(self, T=3, nu=0.0, **kwargs):
        """
        Initializes the WeightedRobustFederatedAveraging instance.

        Args:
            T (int, optional, default=3): Number of iterations for the smoothed Weiszfeld algorithm.
            nu (float, optional, default=0.0): Smoothing parameter to avoid division by zero in distance computation.
        """
        super().__init__(T, nu, **kwargs)

    def __call__(self, gradients, weights=None, *args, **kwargs):
        """
        Aggregates gradients using the smoothed Weiszfeld algorithm with optional weights.

        Args:
            gradients (list): A list of gradient tensors from different clients.
            weights (torch.Tensor, optional, default=None): A tensor representing the weights for each worker.
                                                            If not provided, uniform weights are used.
        Returns:
            torch.Tensor: The aggregated gradient tensor computed using the smoothed Weiszfeld algorithm.
        """
        # Use uniform weights if none are provided
        if weights is None:
            alphas = [1 / len(gradients) for _ in range(len(gradients))]
        else:
            alphas = weights / weights.sum()

        # Initialize z with the same shape as an individual gradient
        z = torch.zeros_like(gradients[0])

        # Apply the smoothed Weiszfeld algorithm to compute the robust mean
        return self.smoothed_weiszfeld(gradients, alphas, z=z, nu=self.nu, T=self.T)

