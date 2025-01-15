from torch.optim import SGD
from .anytime_sgd import AnyTimeSGD


OPTIMIZER_REGISTRY = {
    'sgd': SGD,
    'momentum': SGD,
    'anytime_sgd': AnyTimeSGD,
}


