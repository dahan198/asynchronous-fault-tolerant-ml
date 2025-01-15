from .conv_mnist import ConvNetMNIST
from .conv_cifar10 import ConvNetCIFAR10


MODEL_REGISTRY = {
    'conv_mnist': ConvNetMNIST,
    'conv_cifar10': ConvNetCIFAR10
}
