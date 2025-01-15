import torch
import numpy as np
import random
import inspect


def set_seed(seed):
    """
    Sets the random seed for reproducibility across various libraries and frameworks.

    Args:
        seed (int): The seed value to be set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior in cuDNN for reproducibility
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed for all available CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def filter_valid_args(object_class, **kwargs):
    """
    Filters keyword arguments to retain only those that match the constructor's parameters
    of a given class.

    Args:
        object_class (type): The class whose constructor's parameters are used for filtering.
        **kwargs: Arbitrary keyword arguments to be filtered.

    Returns:
        dict: A dictionary containing only the valid keyword arguments.
    """
    init_signature = inspect.signature(object_class.__init__)
    valid_params = set(init_signature.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return filtered_kwargs


def get_device():
    """
    Returns the appropriate computing device (CPU, CUDA, or MPS) for PyTorch operations.

    Returns:
        torch.device: The available device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')