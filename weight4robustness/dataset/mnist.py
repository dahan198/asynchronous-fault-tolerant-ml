import torchvision
import torchvision.transforms as transforms


class MNIST:
    """
    A utility class for loading the MNIST dataset with standard transformations applied to both
    the training and testing sets.

    The dataset is automatically downloaded if not present in the specified root directory.
    """

    def __init__(self):
        """
        Initializes the MNIST class by defining the data transformations and loading
        the training and testing datasets.

        Transformation applied:
        - Convert images to PyTorch tensors.
        - Normalize images using the mean and standard deviation of the MNIST dataset.
        """
        super().__init__()

        # Define transformation for both training and testing sets
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # Normalize with MNIST statistics
        ])

        # Load the MNIST training set
        self.trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)

        # Load the MNIST testing set
        self.testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
