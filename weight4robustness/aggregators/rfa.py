import torch
from .aggregator import Aggregator


class RobustFederatedAveraging(Aggregator):
    """
    Implements a robust federated averaging algorithm using a smoothed Weiszfeld method
    for gradient aggregation. This approach mitigates the impact of outliers by
    iteratively computing a robust mean based on weighted distances.
    """

    def __init__(self, T=3, nu=0.0, **kwargs):
        """
        Initializes the RobustFederatedAveraging instance.

        Args:
            T (int, optional, default=3): Number of iterations for the smoothed Weiszfeld algorithm.
            nu (float, optional, default=0.0): Smoothing parameter to avoid division by zero in distance computation.
        """
        self.T = T
        self.nu = nu

    def __call__(self, gradients, *args, **kwargs):
        """
        Aggregates the provided gradients using the smoothed Weiszfeld algorithm.

        Args:
            gradients (list): A list of gradient tensors from different clients.

        Returns:
            torch.Tensor: The aggregated gradient tensor.
        """
        alphas = [1 / len(gradients) for _ in range(len(gradients))]
        z = torch.zeros_like(gradients[0])
        return self.smoothed_weiszfeld(gradients, alphas, z=z, nu=self.nu, T=self.T)

    @staticmethod
    def _compute_euclidean_distance(v1, v2):
        """
        Computes the Euclidean distance between two vectors.

        Args:
            v1, v2 (torch.Tensor): Input tensors for which the Euclidean distance is computed.

        Returns:
            float: The Euclidean distance between v1 and v2.
        """
        return (v1 - v2).norm()

    def smoothed_weiszfeld(self, vector, alphas, z, nu, T):
        """
        Applies the smoothed Weiszfeld algorithm to compute a robust mean of the input vectors.

        Args:
            vector (list): List of input vectors to aggregate.
            alphas (list): List of weights associated with each input vector.
            z (torch.Tensor): Initial estimate for the robust mean.
            nu (float): Smoothing parameter to avoid division by zero in distance computation.
            T (int): Number of iterations for the Weiszfeld algorithm.

        Returns:
            torch.Tensor: The robust mean computed after T iterations.
        """
        m = len(vector)
        if len(alphas) != m:
            raise ValueError("The length of alphas must match the number of input vectors.")

        if nu < 0:
            raise ValueError("The smoothing parameter nu must be non-negative.")

        for t in range(T):
            betas = []
            for k in range(m):
                distance = self._compute_euclidean_distance(z, vector[k])
                betas.append(alphas[k] / max(distance, nu))

            z = torch.zeros_like(vector[0])
            for w, beta in zip(vector, betas):
                z += w * beta
            z /= sum(betas)

        return z
