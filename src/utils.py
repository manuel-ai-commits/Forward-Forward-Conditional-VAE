import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src import ff_mnist, FFCCVAE_model
import wandb

from sklearn.decomposition import PCA



def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt

def get_input_layer_size(opt):
    if opt.input.dataset == "mnist":
        return 784  # 28x28 grayscale images
    elif opt.input.dataset == "senti":
        return 302  # Example feature size for a sentiment analysis dataset
    elif opt.input.dataset == "cifar10" or opt.input.dataset == "cifar100" or opt.input.dataset == "GTSRB":
        return 3072  # 32x32 RGB images
    else:
        raise ValueError("Unknown dataset.")

def get_model_and_optimizer(opt):
    model = FFCCVAE_model.FFCCVAE(opt)
    # model = FFCCVAE_model_classifiers.FFCCVAE(opt)

    if "cuda" in opt.device or "mps" in opt.device:
        model = model.to(opt.device)
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
    ]

    if opt.training.optimizer == "SGD":
        
        optimizer = torch.optim.SGD(
            [
                {
                    "params": main_model_params,
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "momentum": opt.training.momentum,
                }
            ]
        )
    elif opt.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": main_model_params,
                    "lr": opt.training.learning_rate,
                    "weight_decay": opt.training.weight_decay,
                    "betas": (opt.training.betas[0] , opt.training.betas[1]), 
                }
            ]
        )
    return model, optimizer

def get_data(opt, partition):
    if opt.input.dataset == "mnist":
        dataset = ff_mnist.MNIST_(opt, partition, number_samples=opt.input.number_samples, preload=False)
    elif opt.input.dataset == "cifar10":
        dataset = ff_mnist.CIFAR_(opt, partition, number_samples=opt.input.number_samples, preload=False)
    elif opt.input.dataset == "fmnist":
        dataset = ff_mnist.FashionMNIST_(opt, partition, number_samples=opt.input.number_samples, preload=False)
    elif opt.input.dataset == "GTSRB":
        dataset = ff_mnist.GTSRB_(opt, partition, allowed_categories=opt.input.classes_allowed, number_samples=opt.input.number_samples, preload=False)
    elif opt.input.dataset == "flowers":
        dataset = ff_mnist.Flowers_(opt,  partition, allowed_categories=opt.input.classes_allowed, number_samples=opt.input.number_samples, preload=False)  
    elif opt.input.dataset == "stl":
        dataset = ff_mnist.STL_(opt, partition, number_samples=opt.input.number_samples, preload=False)
    elif opt.input.dataset == "oxfordpets":
        dataset = ff_mnist.OxfordPets_(opt, partition, allowed_categories=opt.input.classes_allowed, number_samples=opt.input.number_samples, preload=False)
    else:
        raise ValueError("Unknown dataset.")

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=7,
        persistent_workers=True
    )



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dict_to_cuda(opt, obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = value.to(opt.device, non_blocking=True)
    else:
        obj = obj.to(opt.device, non_blocking=True)
    return obj


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device or "mps" in opt.device:
        inputs = dict_to_cuda(opt, inputs)
        labels = dict_to_cuda(opt, labels)
    return inputs, labels
 
# cools down after the first half of the epochs
def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size

def get_mse_cwc(input_tensor, n_classes):
    """
    Computes the mean squared value for each subset of the input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        n_classes (int): Number of classes (to divide the channel dimension into subsets).
        subset_dims (list): Dimensions along which to compute the mean squared value (e.g., [2, 3, 4]).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, n_classes) containing the mean squared values per subset.
    """
    # Step 1: Reshape the tensor to divide the channels into `n_classes` subsets
    reshaped = input_tensor.view(
        input_tensor.shape[0], 
        n_classes, 
        input_tensor.shape[1] // n_classes, 
        input_tensor.shape[2],
        input_tensor.shape[3]
    )
    
    # Step 2: Compute the mean squared value for each subset along the specified dimensions
    mse_per_subset = (reshaped ** 2).mean(dim=[2,3,4])
    
    return mse_per_subset


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value
    wandb.log(partition_scalar_outputs, step=epoch)

# create save_model function
def save_model(model):
    torch.save(model.state_dict(), f"{wandb.run.name}-model.pt")
    # log model to wandb
    wandb.save(f"{wandb.run.name}-model.pt")


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict

def overlay_y_on_x3d(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    with torch.no_grad():

        B, C, H, W = x.shape
        unflatten = nn.Unflatten(1, torch.Size([C, H, W]))

        x_ = x.clone()
        # x_ -> [batch_size, channels * height * width]
        x_ = x_.reshape(x_.size(0), -1)

        x_[:, :10] *= 0.0
        # This line sets the value at the index specified by the label [y] for each sample in the batch
        # to the maximum value in the entire input tensor x_. This is a quick and cheap way to
        # inject a one-hot-like label encoding into the data by setting the position y to a distinct value.
        # x_ -> [batch_size, channels * height * width]
        x_[range(x_.shape[0]), y] = x_.max()

        # x_ -> [batch_size, channels, height, width]
        x_ = unflatten(x_)
        return x_


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    with torch.no_grad():
        x_ = x.clone()
        x_[:, :10] *= 0.0
        batch_range = range(x.shape[0])
        x_[batch_range, y] = x.max()

        return x_


def overlay_y_on_x4d(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    with torch.no_grad():
        B, C, H, W = x.shape

        unflatten = nn.Unflatten(1, torch.Size([H, W]))

        x_ = x.clone()
        for ch in range(C):
            channel = x_[:, ch]
            # print(channel.shape)
            channel = channel.reshape(channel.size(0), -1)
            channel[:, :10] *= 0.0
            channel[range(channel.shape[0]), y] = channel.max()
            channel = unflatten(channel)
            # print(channel.shape)
            x_[:, ch, :, :] = channel

        return x_


#region: 1) Used for visualization of the input and reconstructed images
def visualize_autoencoder_results(inputs, outputs=None, num_images=5, figsize=(15, 15), title="Generated Images", save=True):
    """
    Visualizes input and output images from an autoencoder side-by-side or generated images.
    
    Parameters:
        inputs (torch.Tensor or np.ndarray): The input images, of shape (N, H, W) or (N, C, H, W).
        outputs (torch.Tensor or np.ndarray, optional): The output images, same shape as inputs. Set to None for generated-only visualization.
        num_images (int): Number of images to display in the grid.
        figsize (tuple): Size of the matplotlib figure.
        title (str): Title for generated-only images.
        save (bool): Whether to save the plot as a file.
    
    Returns:
        None
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    if outputs is not None and isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    
    # Handle grayscale images (N, H, W) or (N, C, H, W)
    if inputs.ndim == 4:  # (N, C, H, W) - CIFAR-10 (3, 32, 32)
        inputs = np.transpose(inputs, (0, 2, 3, 1))  # Convert to (N, H, W, C)
        if outputs is not None:
            outputs = np.transpose(outputs, (0, 2, 3, 1))  # Convert to (N, H, W, C)

    # Clip values for visualization if needed
    inputs = np.clip(inputs, 0, 1)  # Assuming normalized inputs
    if outputs is not None:
        outputs = np.clip(outputs, 0, 1)  # Assuming normalized outputs

    # Select random indices for visualization
    indices = np.random.choice(inputs.shape[0], num_images, replace=False)

    # If outputs are provided, compare inputs and outputs side-by-side
    if outputs is not None:
        fig, axes = plt.subplots(2, num_images, figsize=figsize)
        for i, idx in enumerate(indices):
            # Input images
            axes[0, i].imshow(inputs[idx])  # Automatically handles RGB
            axes[0, i].axis('off')
            axes[0, i].set_title("Input")

            # Output images
            axes[1, i].imshow(outputs[idx])  # Automatically handles RGB
            axes[1, i].axis('off')
            axes[1, i].set_title("Output")

    # If outputs are not provided, visualize generated images only
    else:
        fig, axes = plt.subplots(1, num_images, figsize=figsize)
        for i, idx in enumerate(indices):
            axes[i].imshow(inputs[idx])  # Automatically handles RGB
            axes[i].axis('off')
            axes[i].set_title(title)
    
    # Adjust layout
    plt.tight_layout()

    # Save the plot if requested
    if save:
        os.makedirs("results", exist_ok=True)  # Ensure the 'results' directory exists
        plt.savefig(f"results/{title}.png", bbox_inches="tight")
    
    # Show the plot
    plt.show()
#endregion


#region: 2) Display some images
def display_and_save_batch(title, batch, save=True, display=True):
    """
    Display and save a batch of images using matplotlib.
    Parameters:
        - title: Title of the plot and filename for saving.
        - batch: Tensor of images (B, C, H, W).
        - save: Boolean to save the image.
        - display: Boolean to display the image.
    """
    # Convert the batch to a grid
    im = torchvision.utils.make_grid(batch, nrow=int(batch.shape[0]**0.5))
    plt.figure(figsize=(8, 8))
    plt.title(title)
    # Check if the image has 3 channels (RGB), and display it accordingly
    if batch.shape[1] == 3:
        plt.imshow(np.transpose(im.cpu().numpy(), (1, 2, 0)))  # RGB images (3 channels)
    else:
        plt.imshow(np.transpose(im.cpu().numpy(), (1, 2, 0)), cmap='gray')  # Grayscale images
    if save:
        os.makedirs("results", exist_ok=True)
        plt.savefig(f'results/{title}.png', transparent=True, bbox_inches='tight')
    if display:
        plt.show()
    plt.close()
#endregion


#region: 3) Display the latent space conditioning images with 2D PCA
def generate_and_visualize(decoder, device, n_classes=10, num_images=100, latent_dim=100, grid_size=10):
    """
    Generate images conditioned on class labels and visualize them on a 2D latent space grid.
    
    Parameters:
        decoder: The decoder model function
        device: The device (CPU or GPU)
        n_classes: Number of classes
        num_images: Number of images to generate per class
        latent_dim: Dimensionality of the latent space
        grid_size: Unused parameter (kept for compatibility)
    """
    for i in range(n_classes):
        latents = torch.randn(num_images, latent_dim, device=device)
        labels = torch.zeros(num_images, n_classes, device=device)
        labels[:, i] = 1
        
        with torch.no_grad():
            images = decoder(torch.cat((latents, labels), dim=1))
        
        # Reduce latent space to 2D using PCA
        if latent_dim > 2:
            latents_2d = PCA(n_components=2).fit_transform(latents.cpu().numpy())
        else:
            latents_2d = latents.cpu().numpy()
        
        display_image_sparse(latents_2d, images, i)

def display_image_sparse(latents_2d, images, label, title="Generated Images", save=True, display=True, threshold=0.1, color=(0.6, 0.8, 1)):
    """
    Display generated images positioned according to their 2D latent space coordinates.
    
    Parameters:
        latents_2d: 2D coordinates [num_images, 2] for positioning images
        images: Generated images [num_images, C, H, W] (tensor or numpy array)
        label: Class label for the title
        title: Plot title prefix
        save: Whether to save the figure
        display: Whether to display the figure
        threshold: Pixel intensity threshold for transparency (grayscale only)
        color: RGB color tuple for grayscale images
    """
    # Convert to numpy and ensure correct shape
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    
    num_images, C, H, W = images.shape
    
    # Normalize images to [0, 1]
    img_min, img_max = images.min(), images.max()
    if img_max > img_min:
        images = (images - img_min) / (img_max - img_min)
    else:
        images = np.full_like(images, 0.5)
    images = np.clip(images, 0, 1)
    
    # Setup canvas
    padding = 50
    canvas_size = int(np.ceil(np.sqrt(num_images))) * (max(H, W) + 5)
    total_size = canvas_size + 2 * padding
    canvas = np.ones((total_size, total_size, 4 if C == 1 else 3))
    
    # Process images: convert to RGBA for grayscale, transpose for RGB
    if C == 1:
        images_2d = images[:, 0, :, :]
        images_rgba = np.zeros((num_images, H, W, 4))
        for c in range(3):
            images_rgba[..., c] = images_2d * color[c]
        images_rgba[..., 3] = np.where(images_2d < threshold, 0, 1)
        images = images_rgba
    else:
        images = np.transpose(images, (0, 2, 3, 1))
    
    # Normalize latent coordinates and compute positions
    if isinstance(latents_2d, torch.Tensor):
        latents_2d = latents_2d.cpu().numpy()
    latents_2d = np.array(latents_2d)
    
    latents_min = latents_2d.min(axis=0)
    latents_max = latents_2d.max(axis=0)
    latents_range = latents_max - latents_min
    latents_range[latents_range == 0] = 1.0
    latents_2d = (latents_2d - latents_min) / latents_range
    
    positions = (latents_2d * (canvas_size - max(H, W) - 5)).astype(int) + padding
    positions[:, 0] = np.clip(positions[:, 0], padding, total_size - W - padding)
    positions[:, 1] = np.clip(positions[:, 1], padding, total_size - H - padding)
    
    # Place images on canvas with alpha blending
    for i, (x, y) in enumerate(positions):
        x, y = int(x), int(y)
        if x + W > total_size or y + H > total_size or x < 0 or y < 0:
            continue
        
        if C == 1:
            img = images[i]
            canvas_slice = canvas[y:y+H, x:x+W, :].copy()
            alpha = img[:, :, 3:4]
            canvas[y:y+H, x:x+W, :3] = img[:, :, :3] * alpha + canvas_slice[:, :, :3] * (1 - alpha)
            canvas[y:y+H, x:x+W, 3] = np.maximum(canvas_slice[:, :, 3], alpha[:, :, 0])
        else:
            canvas[y:y+H, x:x+W, :] = np.clip(images[i], 0, 1)
    
    # Create plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
    ax.imshow(np.clip(canvas, 0, 1), interpolation='nearest')
    
    # Set axis labels and ticks
    original_x = latents_2d[:, 0] * (latents_max[0] - latents_min[0]) + latents_min[0]
    original_y = latents_2d[:, 1] * (latents_max[1] - latents_min[1]) + latents_min[1]
    
    ax.set_xticks(np.linspace(padding, canvas_size + padding, 5))
    ax.set_yticks(np.linspace(padding, canvas_size + padding, 5))
    ax.set_xticklabels(np.round(np.linspace(original_x.min(), original_x.max(), 5), 2))
    ax.set_yticklabels(np.round(np.linspace(original_y.min(), original_y.max(), 5), 2))
    
    ax.set_xlabel('Latent Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latent Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} - Class {label}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    if save:
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/conditioned_{label}.png", dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
    
    if display:
        plt.show()
    plt.close()
#endregion

#region: 4) Display the latent space conditioning images with 1D PCA in rows
def generate_and_visualize_1D(decoder, device, class_names=None, n_classes=10, num_images=100, latent_dim=100, grid_size=10):
    """
    Generate images conditioned on a fixed class label and visualize them on a 2D grid.

    Parameters:
        - decoder: The decoder model.
        - device: The device (CPU or GPU).
        - n_classes: Number of classes.
        - num_images: Number of images to generate per class.
        - latent_dim: Dimensionality of the latent space.
        - grid_size: Number of images per row.
    """

    all_latents = []
    all_images = []
    all_labels = []

    # Generate images for each class
    for class_idx in range(n_classes):
        latents = torch.randn(num_images, latent_dim, device=device)

        labels = torch.zeros(num_images, n_classes, device=device)  # One-hot encoding
        labels[:, class_idx] = 1  # Set the class-specific index to 1

        # Concatenate latent vectors with class labels
        conditional_inputs = torch.cat((latents, labels), dim=1)

        # Generate images
        with torch.no_grad():
            images = decoder(conditional_inputs)

        # Apply PCA to reduce latent space to 1D
        pca = PCA(n_components=1)
        latents_1D = pca.fit_transform(latents.cpu().numpy())  # Reduce to 1D

        # Store results
        all_latents.append(latents_1D)
        all_images.append(images.cpu())  # Move images to CPU for processing
        all_labels.append(np.full(num_images, class_idx))  # Store labels

    # Convert lists to arrays
    all_latents = np.concatenate(all_latents, axis=0)
    all_images = torch.cat(all_images, dim=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Visualize all images together
    display_image_rows("PLOT", all_images, all_labels, all_latents,  grid_size, class_names)


def display_image_rows(title, images, labels, all_latents, grid_size=5, class_names=None, save=True, display=True):
    """
    Display and save images in a grid by their class names instead of raw labels, with labels centered vertically on each row.

    Parameters:
        - title: Title of the plot and filename for saving.
        - images: Tensor of images (B, C, H, W), where C can be 1 (grayscale) or 3 (RGB).
        - labels: Labels for the images.
        - all_latents: Latent values corresponding to the images (B, L).
        - grid_size: Number of images per row.
        - class_names: List of class names corresponding to label indices.
        - save: Boolean to save the image.
        - display: Boolean to display the image.
    """
    # Ensure labels is a torch tensor
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    # Ensure latents are a numpy array
    if isinstance(all_latents, torch.Tensor):
        all_latents = all_latents.numpy()

    # Sort images, labels, and latents based on their latent values
    sorted_indices = np.argsort(all_latents[:, 0])  # Sort by first latent component
    images = images[sorted_indices]
    labels = labels[sorted_indices]

    # Create a grid for each class
    unique_classes = torch.unique(labels)
    
    all_images = []
    all_labels = []
    for cls in unique_classes:
        class_images = images[labels == cls][:grid_size]  # Keep only up to grid_size images per class
        all_images.append(class_images)
        all_labels.append([cls.item()] * class_images.size(0))

    # Flatten lists
    all_images = torch.cat(all_images, dim=0)
    all_labels = np.concatenate(all_labels)

    # Ensure images are in range [0, 1]
    all_images = (all_images - all_images.min()) / (all_images.max() - all_images.min())

    # Create a grid of images
    grid = torchvision.utils.make_grid(all_images, nrow=grid_size, padding=2, pad_value=1)
    grid_np = np.transpose(grid.cpu().numpy(), (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(grid_np)
    ax.axis('off')

    # Display class names instead of raw labels
    num_rows = len(all_labels) // grid_size
    for i in range(num_rows):
        label_idx = all_labels[i * grid_size]
        class_name = class_names[label_idx] if class_names else f'Class {label_idx}'
        label_y_pos = (i * grid_np.shape[0]) / num_rows + grid_np.shape[0] // (2 * num_rows)
        ax.text(-grid_np.shape[1] // grid_size, label_y_pos, class_name,
                color='black', fontsize=12, verticalalignment='center', horizontalalignment='center')

    # Set title
    plt.title(title)

    # Save if needed
    if save:
        os.makedirs("results", exist_ok=True)
        plt.savefig(f'results/{title}.png', transparent=True, bbox_inches='tight')
    
    # Display if needed
    if display:
        plt.show()

    plt.close()

#endregion


#region: 5) Visualize the latent space points
def visualize_latent_space(z, labels, class_names, latent_dim=2, device='cpu', title="Latent Space Visualization", save=True):
    """
    Visualizes the latent space of a trained encoder using PCA if latent_dim > 2.

    Parameters:
        - z: Latent vectors from the encoder.
        - labels: Class labels corresponding to the data.
        - class_names: List of class names.
        - latent_dim: Dimensionality of the latent space.
        - device: Device to run the model on ('cpu' or 'cuda').
        - title: Title of the plot.
        - save: Boolean flag to save the figure.
    """
    
    # Ensure labels are on CPU before using NumPy operations
    labels = labels.cpu().numpy()

    # Reduce dimensions if necessary
    if latent_dim > 2:
        pca = PCA(n_components=2)
        z = pca.fit_transform(z.cpu().numpy())
    else:
        z = z.cpu().numpy()

    # Use a categorical colormap (discrete colors)
    num_classes = len(class_names)
    colormap = plt.cm.get_cmap('tab10', num_classes)  # 'tab10' works well for up to 10 classes. Use 'tab20' for more.
    
    # Assign a color to each class
    unique_labels = np.unique(labels)
    colors = [colormap(i) for i in range(num_classes)]
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Plot the latent space with distinct colors
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        mask = labels == label
        plt.scatter(z[mask, 0], z[mask, 1], color=label_color_map[label], label=class_names[label], alpha=0.7, s=50)

    # Title, labels, and legend
    plt.title(title)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")  # Show legend with class names

    if save:
        os.makedirs("results", exist_ok=True)  # Ensure the 'results' directory exists
        plt.savefig(f"results/latent_space_{title}.png", bbox_inches="tight")

    # Show the plot
    plt.show()
#endregion