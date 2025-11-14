import hydra
from omegaconf import DictConfig
from src import ff_mnist  # Assuming this is where 'parse_args' is defined

import os
import matplotlib.pyplot as plt
import numpy as np

num_classes_dict = {
    "mnist": 10,         # MNIST has 10 classes
    "cifar10": 10,       # CIFAR-10 has 10 classes
    "fmnist": 10,        # FashionMNIST has 10 classes
    "gtsrb": 43          # GTSRB has 43 classes
}


def get_label_distribution(dataset, num_classes):
    """
    Get the distribution of labels in the dataset.
    
    Parameters:
    - dataset: The dataset object.
    - num_classes: Number of distinct classes in the dataset.
    
    Returns:
    - A list with the counts of each label in the dataset.
    """
    label_counts = np.zeros(num_classes, dtype=int)
    
    for _, label in dataset:
        label_counts[label.item()] += 1  # Increment the count of the corresponding label
    
    return label_counts

def plot_label_distribution(label_counts, dataset_name):
    """
    Plot a bar graph for the label distribution and save it to the results folder.
    
    Parameters:
    - label_counts: List with the counts of each label.
    - dataset_name: Name of the dataset for the plot title.
    """
    # Create the results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(label_counts)), label_counts)
    plt.title(f"Label Distribution for {dataset_name}")
    plt.xlabel("Label")
    plt.ylabel("Frequency")
    plt.xticks(range(len(label_counts)))

    # Save the plot in the results folder
    plt.savefig(f"results/{dataset_name}_label_distribution.png", bbox_inches="tight")
    plt.show()
    plt.close()  # Close the plot to avoid memory issues

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(opt: DictConfig): 
    datasets = {
        "mnist": ff_mnist.MNIST_(opt, "train", number_samples=opt.input.number_samples, preload=False),
        "cifar10": ff_mnist.CIFAR_(opt, "train", number_samples=opt.input.number_samples, preload=False),
        "fmnist": ff_mnist.FashionMNIST_(opt, "train", number_samples=opt.input.number_samples, preload=False),
        "gtsrb": ff_mnist.GTSRB_(opt, "train", number_samples=opt.input.number_samples, preload=False)
    }

    for dataset_name, dataset in datasets.items():
        num_classes = num_classes_dict[dataset_name]
        label_counts = get_label_distribution(dataset, num_classes)
        plot_label_distribution(label_counts, dataset_name)
if __name__ == "__main__":
    main()



