from __future__ import print_function

import random

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dataset_utils
import torchvision.transforms as transforms


def set_environment(seed_range, n_gpu):
    # Set random seed for reproducibility
    create_seed(seed_range[0], seed_range[1])
    # Defines the device(s) that will be used
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    return device


def set_transform(img_size, img_depth, norm=0.5):
    transformations = [transforms.Resize(img_size),
                       transforms.CenterCrop(img_size),
                       transforms.ToTensor()]

    if img_depth == 3:
        transformations.append(transforms.Normalize((norm, norm, norm), (norm, norm, norm)))
    elif img_depth == 2:
        transformations.append(transforms.Normalize((norm, norm), (norm, norm)))
    elif img_depth == 1:
        transformations.append(transforms.Normalize((norm,), (norm,)))
    else:
        raise ValueError("Image depth not supported")

    transform = transforms.Compose(transformations)

    return transform


def create_seed(s_min, s_max):
    rand_seed = random.randint(s_min, s_max)
    print("Random Seed: ", rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)


def load_image_dataset(data_root, transform):
    dataset = dataset_utils.ImageFolder(root=data_root, transform=transform)
    return dataset


def load_image_data(data_root, transform, batch_size, norm=1, shuffle=False, data_loader_workers=0):
    # Load dataset
    dataset = load_image_dataset(data_root, transform)
    # Create data loader
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=data_loader_workers)

    return data_loader


def load_base_dataset(dataset_name, data_root, transform, batch_size, norm=1, shuffle=False, data_loader_workers=0):
    if dataset_name == "MNIST":
        dataset = dataset_utils.MNIST(data_root, train=True, download=True, transform=transform)
    elif dataset_name == "Fashion-MNIST":
        dataset = dataset_utils.FashionMNIST(data_root, train=True, download=True, transform=transform)
    elif dataset_name == "KMNIST":
        dataset = dataset_utils.KMNIST(data_root, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"The dataset {dataset_name} is not available")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=data_loader_workers)

    return data_loader
