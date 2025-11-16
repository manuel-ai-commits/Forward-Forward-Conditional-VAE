import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from hydra.utils import get_original_cwd

from torchvision.datasets import GTSRB, MNIST, CIFAR10, CIFAR100, FashionMNIST, Flowers102, Caltech101, STL10, OxfordIIITPet, SVHN
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale, ColorJitter
from torch.utils.data import Dataset, Subset


from src import utils
import torch
import numpy as np
import cv2

"""
    MNIST dataset class
"""
class MNIST_(Dataset):
    def __init__(self, opt, partition, number_samples=None, FF_rep=False, preload=False):
        # Define transformations
        transform = Compose([
            ToTensor()
            # normalize dataset 0,1 mean variance:
            # Normalize((0.1307,), (0.3081,))
        ])
        
        self.partition = partition
        
        # Load the MNIST dataset
        if self.partition == "train":
            dataset = MNIST(
                root=os.path.join(get_original_cwd(), opt.input.path),
                train=True,
                download=True,
                transform=transform
            )
        elif self.partition in ["val", "test"]:
            dataset = MNIST(
                root=os.path.join(get_original_cwd(), opt.input.path),
                train=False,
                download=True,
                transform=transform
            )
        else:
            raise NotImplementedError(f"Partition '{partition}' is not supported. Use 'train' or 'test'.")

        # Subset the dataset if necessary
        dataset_size = len(dataset)
        print(f"Dataset size: {dataset_size}")
        if number_samples is not None:
            number_samples = min(number_samples, dataset_size)
            dataset = Subset(dataset, range(number_samples))

        # Preload data into memory if enabled
        if preload:
            self.data = [dataset[i] for i in range(len(dataset))]
        else:
            self.dataset = dataset

        self.FF_rep = FF_rep
        self.preload = preload

    def __getitem__(self, index):
        if self.preload:
            x, y = self.data[index]
        else:
            x, y = self.dataset[index]

        if self.FF_rep:
            x = x.flatten()

        return x, torch.tensor(y).long()

    def __len__(self):
        if self.preload:
            return len(self.data)
        return len(self.dataset)
    

"""
    FashionMNIST dataset class
"""
class FashionMNIST_(Dataset):
    def __init__(self, opt, partition, number_samples=None, FF_rep=False, preload=False):
        # Define transformations
        transform = Compose([
            ToTensor()
            # Dataset can be normalized to the range [-1, 1]
            #Normalize((0.5,), (0.5,))  # Updated normalization values for FashionMNIST
        ])
        
        self.partition = partition
        
        # Load the FashionMNIST dataset
        if self.partition == "train":
            dataset = FashionMNIST(
                root=os.path.join(get_original_cwd(), opt.input.path),
                train=True,
                download=True,
                transform=transform
            )
        elif self.partition in ["val", "test"]:
            dataset = FashionMNIST(
                root=os.path.join(get_original_cwd(), opt.input.path),
                train=False,
                download=True,
                transform=transform
            )
        else:
            raise NotImplementedError(f"Partition '{partition}' is not supported. Use 'train' or 'test'.")

        # Subset the dataset if necessary
        dataset_size = len(dataset)
        print(f"Dataset size: {dataset_size}")
        if number_samples is not None:
            number_samples = min(number_samples, dataset_size)
            dataset = Subset(dataset, range(number_samples))

        # Preload data into memory if enabled
        if preload:
            self.data = [dataset[i] for i in range(len(dataset))]
        else:
            self.dataset = dataset

        self.FF_rep = FF_rep
        self.preload = preload

    def __getitem__(self, index):
        if self.preload:
            x, y = self.data[index]
        else:
            x, y = self.dataset[index]

        if self.FF_rep:
            x = x.flatten()

        return x, torch.tensor(y).long()

    def __len__(self):
        if self.preload:
            return len(self.data)
        return len(self.dataset)


"""
    SVHN dataset class
"""
class SVHN_(Dataset):
    def __init__(self, opt, partition, number_samples=None, FF_rep=False, preload=False):
        # Define transformations
        transform = Compose([
            ToTensor()
            # to normalize data: 
            # Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN normalization values
        ])
        
        self.partition = partition
        
        # Load the SVHN dataset
        if self.partition == "train":
            dataset = SVHN(
                root=os.path.join(get_original_cwd(), opt.input.path),
                split='train',
                download=True,
                transform=transform
            )
        elif self.partition == "test":
            dataset = SVHN(
                root=os.path.join(get_original_cwd(), opt.input.path),
                split='test',
                download=True,
                transform=transform
            )
        elif self.partition == "extra":
            dataset = SVHN(
                root=os.path.join(get_original_cwd(), opt.input.path),
                split='extra',
                download=True,
                transform=transform
            )
        else:
            raise NotImplementedError(f"Partition '{partition}' is not supported. Use 'train', 'test', or 'extra'.")

        # Subset the dataset if necessary
        dataset_size = len(dataset)
        print(f"Dataset size: {dataset_size}")
        if number_samples is not None:
            number_samples = min(number_samples, dataset_size)
            dataset = Subset(dataset, range(number_samples))

        # Preload data into memory if enabled
        if preload:
            self.data = [dataset[i] for i in range(len(dataset))]
        else:
            self.dataset = dataset

        self.FF_rep = FF_rep
        self.preload = preload

    def __getitem__(self, index):
        if self.preload:
            x, y = self.data[index]
        else:
            x, y = self.dataset[index]

        if self.FF_rep:
            x = x.flatten()

        return x, torch.tensor(y).long()

    def __len__(self):
        if self.preload:
            return len(self.data)
        return len(self.dataset)

"""
    CIFAR dataset class
"""
class CIFAR_(Dataset):
    def __init__(self, opt, partition, dataset_name= "cifar10", number_samples=None, FF_rep=False, preload=False):
        # Define the transformations
        if opt.input.dataset != 'cifar10' and opt.input.dataset != 'cifar100':
            self.dataset_name = dataset_name
        else:
            self.dataset_name = opt.input.dataset
        transform = Compose([
            # Grayscale(num_output_channels=1),
            Resize((64, 64)),
            ToTensor()
            # Normalize(
            #     (0.4914, 0.4822, 0.4465) if self.dataset_name == 'cifar10' else (0.5071, 0.4867, 0.4408),
            #     (0.2023, 0.1994, 0.2010) if self.dataset_name == 'cifar10' else (0.2675, 0.2565, 0.2761)
            # )
        ])
        
        self.partition = partition
        self.preload = preload

        # Load CIFAR-10 or CIFAR-100 based on the dataset parameter
        if self.dataset_name == 'cifar10':
            DatasetClass = CIFAR10
        elif self.dataset_name == 'cifar100':
            DatasetClass = CIFAR100
        else:
            raise ValueError(f"Dataset '{dataset}' is not supported. Use 'cifar10' or 'cifar100'.")

        # Load the dataset based on the partition
        if self.partition == "train":
            self.dataset = DatasetClass(
                root=os.path.join(get_original_cwd(), opt.input.path),
                train=True,
                download=True,
                transform=transform
            )
        elif self.partition in ["val", "test"]:
            self.dataset = DatasetClass(
                root=os.path.join(get_original_cwd(), opt.input.path),
                train=False,
                download=True,
                transform=transform
            )
        else:
            raise NotImplementedError(f"Partition '{partition}' is not supported. Use 'train', 'val', or 'test'.")

        # Adjust the number of samples if specified
        dataset_size = len(self.dataset)
        print(f"Dataset size: {dataset_size}")
        if number_samples is not None:
            number_samples = min(number_samples, dataset_size)
            self.dataset = Subset(self.dataset, range(number_samples))

        self.FF_rep = FF_rep

        # Preload dataset into memory
        if self.preload:
            print("Preloading dataset into memory...")
            self.preloaded_data = [(x, y) for x, y in self.dataset]
        else:
            self.preloaded_data = None

    def __getitem__(self, index):
        if self.preloaded_data is not None:  # If preloaded, use the cached data
            x, y = self.preloaded_data[index]
        else:  # Otherwise, fetch the data dynamically
            x, y = self.dataset[index]

        if self.FF_rep:
            x = x.flatten()
        return x, torch.tensor(y).long()

    def __len__(self):
        return len(self.dataset)


"""
    GTSRB dataset class
"""
class GTSRB_(Dataset):
    def __init__(self, opt, partition, size=(64, 64), number_samples=None, FF_rep=False, allowed_categories=None, preload=False):
        transform = Compose([
            Resize(size),
            ToTensor(),
            ColorJitter(brightness=(1, 2))
            # Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
        ])

        self.partition = partition
        self.preload = preload

        # Load the dataset based on the partition
        if self.partition == "train":
            self.dataset = GTSRB(
                root=os.path.join(get_original_cwd(), opt.input.path),
                split='train',
                download=True,
                transform=transform
            )
        elif self.partition in ["val", "test"]:
            self.dataset = GTSRB(
                root=os.path.join(get_original_cwd(), opt.input.path),
                split='test',
                download=True,
                transform=transform
            )
        else:
            raise NotImplementedError(f"Partition '{partition}' is not supported. Use 'train' or 'test'.")
        

        # Directly filter by allowed categories using `dataset.targets`
        print(f"Dataset size before filtering: {len(self.dataset)}")
        if allowed_categories is not None:
            print("Filtering by allowed categories...")
            labels = torch.tensor([sample[1] for sample in self.dataset._samples])  # Access the labels directly
            mask = torch.isin(labels, torch.tensor(allowed_categories))
            indices = torch.nonzero(mask).squeeze(1).tolist()
            # print(f"labels: {labels}")
            # print(f"Indices: {indices}")
            # print(f"Mask: {mask}")
            self.dataset = Subset(self.dataset, indices)

        # Adjust the number of samples if necessary
        dataset_size = len(self.dataset)
        print(f"Dataset size after filtering: {dataset_size}")
        if number_samples is not None:
            number_samples = min(number_samples, dataset_size)
            self.dataset = Subset(self.dataset, range(number_samples))
        print(f"Dataset size after subsampling: {len(self.dataset)}")

        # Preload data if requested
        if self.preload:
            self._preload_data()

        self.FF_rep = FF_rep

    def _preload_data(self):
        """Preload data into memory efficiently."""
        print("Preloading data into memory...")
        data, labels = [], []

        for x, y in self.dataset:
            data.append(x)
            labels.append(y)

        # Convert lists to tensors for efficient access
        self.data = torch.stack(data)  # Stack images into a single tensor
        self.labels = torch.tensor(labels)  # Convert labels to a tensor

    def __getitem__(self, index):
        if self.preload:
            # Access preloaded data
            x = self.data[index]
            y = self.labels[index]
        else:
            x, y = self.dataset[index]

        if self.FF_rep:
            x = x.flatten()

        return x, torch.tensor(y).long()

    def __len__(self):
        return len(self.dataset)
    
