from __future__ import print_function

import random
from datetime import datetime
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dataset_utils
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


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


def load_model(model, optimizer, save_path):

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model.eval()

    return model, optimizer, epoch


def save_model(model, optimizer, epoch, save_path):

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def plot_loss_resutls(gen_loss, disc_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_loss, label="G")
    plt.plot(disc_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def save_loss_plot(gen_loss, disc_loss, path):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_loss, label="G")
    plt.plot(disc_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{path}/gen_disc_loss.png")


def create_report(r_path, s_date, gen_net, disc_net, optimizer, loss_fn, dataset_name, lr, epochs, disc_loss, gen_loss):
    current_date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
    filename = r_path + f"{current_date}.txt"

    with open(filename, 'w') as file:
        file.write(f"Start date:\t{s_date.strftime('%d-%m-%Y_%I-%M-%S_%p')}\n")
        file.write(f"End date:\t{current_date}\n")
        file.write("------------NET CONFIGURATION------------\n")
        file.write("Generator configuration\n")
        file.write(f"{gen_net}\n")
        file.write("Discriminator configuration\n")
        file.write(f"{disc_net}\n")
        file.write("------------LOSS FN / OPTIMIZER------------\n")
        file.write(f"Optimizer\t{optimizer}\n")
        file.write(f"Loss Function:\t{loss_fn}\n")
        file.write("------------Dataset------------\n")
        file.write(f"Data set name:\t{dataset_name}\n")
        file.write("------------Training------------\n")
        file.write(f"Learning rate:\t{lr}\n")
        file.write(f"Epochs:\t{epochs}\n")
        file.write("------------Results------------\n")
        file.write(f"Generator loss:\t{gen_loss}\n")
        file.write(f"Discriminator loss:\t{disc_loss}\n")
