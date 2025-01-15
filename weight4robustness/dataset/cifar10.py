import torchvision
import torchvision.transforms as transforms


class CIFAR10:
    """
    A utility class for loading the CIFAR-10 dataset with predefined data augmentations
    for training and standard transformations for testing.

    The dataset is automatically downloaded if not already present in the specified root directory.
    """

    def __init__(self):
        """
        Initializes the CIFAR10 class by defining data transformations for training and testing,
        and loading the respective datasets.
        """
        super().__init__()

        # Define data augmentation for the training set
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=2),  # Randomly crop with padding
            transforms.RandomHorizontalFlip(),  # Random horizontal flipping
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010))  # Normalize with CIFAR-10 statistics
        ])

        # Define standard transformation for the testing set
        transform_test = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010))  # Normalize with CIFAR-10 statistics
        ])

        # Load the CIFAR-10 training set
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        # Load the CIFAR-10 testing set
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
