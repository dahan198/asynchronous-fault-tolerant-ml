import torch


class Worker:
    """
    Base class for a worker in a distributed optimization setup.
    This class handles the basic attributes common to all worker types,
    such as momentum and beta (momentum coefficient).
    """

    def __init__(self, beta):
        """
        Initializes the Worker with a specified momentum coefficient.

        Args:
            beta (float): Momentum coefficient for gradient updates.
        """
        self.beta = beta
        self.momentum = None
        self.two_passes = False  # Indicates whether two passes are needed for the worker
        self.current_sample = None
        self.next_sample = None

    def step(self, gradients):
        """
        Placeholder method to be implemented by derived classes.

        Args:
            gradients (torch.Tensor): Gradients from the current step.

        Returns:
            torch.Tensor: Updated gradients or momentum.
        """
        raise NotImplementedError("The 'step' method must be implemented by subclasses.")


class WorkerSGD(Worker):
    """
    Implements a simple Stochastic Gradient Descent (SGD) worker without momentum.
    """

    def __init__(self, beta=None):
        """
        Initializes the WorkerSGD class.

        Args:
            beta (float): Momentum coefficient (unused in this worker type).
        """
        super().__init__(beta)
        self.momentum = None

    def step(self, gradients):
        """
        Updates the gradients without applying momentum.

        Args:
            gradients (torch.Tensor): Gradients from the current step.

        Returns:
            torch.Tensor: The same gradients, as no momentum is applied.
        """
        self.momentum = gradients.clone()
        return gradients


class WorkerMomentum(Worker):
    """
    Implements a worker that applies momentum-based updates.
    """

    def __init__(self, beta):
        """
        Initializes the WorkerMomentum class with a specified momentum coefficient.

        Args:
            beta (float): Momentum coefficient for gradient updates.
        """
        super().__init__(beta)

    def step(self, gradients):
        """
        Applies momentum-based updates to the gradients.

        Args:
            gradients (torch.Tensor): Gradients from the current step.

        Returns:
            torch.Tensor: Updated gradients with momentum applied.
        """
        if self.momentum is None:
            self.momentum = gradients.clone()
        else:
            self.momentum.mul_(self.beta)  # Scale momentum by beta (in-place)
            self.momentum.add_(gradients, alpha=1 - self.beta)  # Add scaled gradients (in-place)
        return self.momentum


class WorkerSTORM(Worker):
    """
    Implements a worker for the STORM optimization algorithm, which adapts its updates
    based on a running estimator of the gradient.
    """

    def __init__(self, beta):
        """
        Initializes the WorkerSTORM class.

        Args:
            beta (float): Initial momentum coefficient.
        """
        super().__init__(beta)
        self.g_tilde = None  # Running estimator of the gradient
        self.two_passes = True  # Indicates that two passes are needed for this worker

    def step(self, gradients):
        """
        Computes the STORM estimator by applying fixed momentum coefficient.

        Args:
            gradients (torch.Tensor): Gradients from the current step.

        Returns:
            torch.Tensor: The current momentum (not updated in this step).
        """
        if self.momentum is None:
            self.momentum = gradients.clone()
            self.g_tilde = gradients.clone()
        else:
            difference = self.momentum.sub(self.g_tilde)  # difference = self.momentum - self.g_tilde
            difference.mul_(1 - self.beta)  # Scale difference by (1 - beta) in-place
            self.momentum.copy_(gradients)  # Copy gradients to momentum
            self.momentum.add_(difference)  # Add scaled difference to momentum

        return self.momentum

    def update_gtilde(self, gradients):
        """
        Updates g_tilde.

        Args:
            gradients (torch.Tensor): Gradients from the current step.
        """
        self.g_tilde = gradients.clone()
