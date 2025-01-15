class Aggregator:
    """
    A base class for implementing aggregation methods in federated learning.

    This class defines the interface for aggregating gradients from multiple clients,
    with optional weighting for the contribution of each client.
    """

    def __call__(self, gradients, weights=None, *args, **kwargs):
        """
        Aggregates gradients from multiple sources, optionally using a set of weights.

        Args:
            gradients (list or tensor): A collection of gradients to be aggregated.
            weights (list or tensor, optional): A collection of weights corresponding to the gradients.
                                                If provided, the gradients will be aggregated as a weighted sum.
                                                If `None`, equal weighting is assumed.

        Returns:
            aggregated_gradient (tensor): The resulting aggregated gradient.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")
