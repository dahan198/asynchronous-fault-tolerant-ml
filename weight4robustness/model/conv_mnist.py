import torch.nn as nn
import torch.nn.functional as F


class ConvNetMNIST(nn.Module):
    """
    A convolutional neural network (CNN) designed for the MNIST dataset.
    The network consists of two convolutional layers followed by two fully connected layers,
    with batch normalization applied before the final classification layer.
    """

    def __init__(self):
        """
        Initializes the ConvNetMNIST model with:
        - Two convolutional layers
        - One batch normalization layer
        - Two fully connected layers
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)  # Grayscale input
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=800, out_features=50)  # Input size adjusted for flattened output
        self.bn = nn.BatchNorm1d(num_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)  # 10 output classes for MNIST digits

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.bn(self.fc1(x)))
        x = self.fc2(x)
        return x
