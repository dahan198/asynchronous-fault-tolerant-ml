import torch.nn as nn
import torch.nn.functional as F


class ConvNetCIFAR10(nn.Module):
    """
    A convolutional neural network (CNN) designed for the CIFAR-10 dataset.
    This network consists of two convolutional layers followed by two fully connected layers.
    Batch normalization is applied before the final output layer.
    """

    def __init__(self):
        """
        Initializes the ConvNetCIFAR10 model with:
        - Two convolutional layers
        - One batch normalization layer
        - Two fully connected layers
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1)  # RGB input
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=1250, out_features=50)  # Adjusted input size for fully connected layer
        self.bn = nn.BatchNorm1d(num_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)

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
