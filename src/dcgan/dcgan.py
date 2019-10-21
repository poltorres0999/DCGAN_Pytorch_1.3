from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dataset_utils
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import argparse
import configparser

from src.dcgan.generator import Generator
from src.dcgan.discriminator import Discriminator

DEFAULT_CONFIG_PATH = "../../res/default_config.ini"


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', help="Path to configuration file '.ini.")
    parser.add_argument('-lm', '--load_models', action='store_true', help="If active, loads the specified models (config.ini)")
    parser.add_argument('-d', '--enable_debug', action='store_true', help="Enable debug option")
    parser.add_argument('-plt', '--enable_plot', action='store_true', help="Plots will be shown during execution")

    return parser.parse_args()


def process_args(args):
    config_path = DEFAULT_CONFIG_PATH
    load_models = False
    debug_enabled = False
    plot_enabled = False

    if args.config_path:
        config_path = args.config_path
    if args.load_models:
        load_models = True
    if args.enable_debug is not None:
        debug_enabled = args.enable_debug
    if args.enable_plot is not None:
        plot_enabled = args.enable_plot

    return config_path, load_models, debug_enabled, plot_enabled


def load_config_file(file_path):
    try:
        conf = configparser.ConfigParser()
        conf.read(file_path)
        return conf
    except configparser.Error as e:
        raise Exception(f"Error while parsing configuration file {file_path}, error: {e}")


def create_seed(s_min, s_max):
    rand_seed = random.randint(s_min, s_max)
    print("Random Seed: ", rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)


def load_image_dataset(data_root, image_size, norm=0.5):
    dataset = dataset_utils.ImageFolder(root=data_root,
                                        transform=transforms.Compose([
                                            transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((norm, norm, norm), (norm, norm, norm))
                                        ]))
    return dataset


def create_data_loader(dataset, shuffle=True, batch_size=1, num_workers=0):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    return data_loader


def plot_images(img_data, num_images, fig_size, device, plt_title=""):
    plt.figure(figsize=(fig_size, fig_size))
    plt.axis("off")
    plt.title(plt_title)
    plt.imshow(np.transpose(
        vutils.make_grid(img_data.to(device)[:num_images], padding=2, normalize=True).cpu(), (1, 2, 0)))


def plot_loss_results(disc_loss, gen_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_loss, label="Gen")
    plt.plot(disc_loss, label="Disc")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_gen_loss(gen_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator Loss During Training")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_disc_loss(disc_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Loss During Training")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def set_environment(seed_range, n_gpu):
    # Set random seed for reproducibility
    create_seed(seed_range[0], seed_range[1])
    # Defines the device(s) that will be used
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    return device


def load_data(data_root, img_size, batch_size, norm=1, shuffle=False, data_loader_workers=0):
    # Load dataset
    dataset = load_image_dataset(data_root, img_size, norm)
    # Create data loader
    data_loader = create_data_loader(dataset, True, batch_size, data_loader_workers)

    return data_loader


def show_generated_images(num_images_x, num_images_y, input_images):
    fig = plt.figure(figsize=(num_images_x, num_images_y))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in input_images]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())


def create_bce_loss_optimizers(discriminator, generator, learning_rate, beta):
    # Initialize BCELoss function
    loss_fn = nn.BCELoss()
    # Setup Adam optimizers for both G and D
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta, 0.999))
    gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta, 0.999))

    return loss_fn, gen_optimizer, disc_optimizer


def optimize_discriminator_net(discriminator_net, loss_fn, real_data, disc_optimizer, fake_data,
                               real_labels, fake_labels):
    ############################
    # Update Discriminator network: maximize log(D(G(z)))
    ###########################
    # -------- Train discriminator with all-real batch ------------
    # Set discriminator gradient values to 0
    discriminator_net.zero_grad()
    # Forward pass real batch through D
    # view(-1) -> Automatically re-shapes
    output = discriminator_net(real_data).view(-1)
    # Calculate loss on all-real batch
    disc_err_real = loss_fn(output, real_labels)
    # Calculate gradients for D in backward pass
    disc_err_real.backward()
    D_x = output.mean().item()

    # ------- Train with all-fake batch -------
    # Classify all fake batch with Discriminator
    output = discriminator_net(fake_data.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    disc_err_fake = loss_fn(output, fake_labels)
    # Calculate the gradients for this batch
    disc_err_fake.backward()
    D_G_z1 = output.mean().item()
    # Add the gradients from the all-real and all-fake batches
    disc_error = disc_err_real + disc_err_fake
    # Update discriminator with the calculated gradients
    disc_optimizer.step()

    return D_x, D_G_z1, disc_error


def optimize_generator_net(generator_net, discriminator_net, loss_fn, gen_optimizer, fake_data, labels):
    ############################
    # Update Generator network: maximize log(D(G(z)))
    ###########################
    generator_net.zero_grad()
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = discriminator_net(fake_data).view(-1)
    # Calculate Generator's loss based on Discriminator's output
    gen_error = loss_fn(output, labels)
    # Calculate gradients for Generator
    gen_error.backward()
    D_G_z2 = output.mean().item()
    # Update G
    gen_optimizer.step()

    return gen_error, D_G_z2


def train(device, data_loader, generator_net, discriminator_net, loss_fn, disc_optimizer, gen_optimizer,
          num_epochs=1, noise_v_size=100):
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    # Training Loop
    # Lists to keep track of progress
    img_list = []
    gen_loss = []
    disc_loss = []
    iters = 0
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, noise_v_size, 1, 1, device=device)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        print(f"Started epoch [{epoch}] / [{num_epochs}]")
        # For each batch in the data_loader
        # Note -> enumerate outputs (int index, item from iterable object)
        for i, train_data in enumerate(data_loader):
            print(f"Iteration [{i}] / [{len(data_loader)}]")
            # ---Prepare data for the current training step---
            real_data = train_data[0].to(device)
            # Get batch size
            batch_size = real_data.size(0)
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, noise_v_size, 1, 1, device=device)
            # Generate fake image batch with Generator
            fake_data = generator_net(noise)
            # Create labels
            real_labels = torch.full((batch_size,), real_label, device=device)
            fake_labels = torch.full((batch_size,), fake_label, device=device)

            disc_error, dr_o_x, dr_g_o_z1 = optimize_discriminator_net(discriminator_net, loss_fn, real_data,
                                                                       disc_optimizer, fake_data, real_labels,
                                                                       fake_labels)
            # fake labels are real for generator cost (maximize log(D(G(z))))
            gen_error, dr_g_o_z2 = optimize_generator_net(generator_net, discriminator_net, loss_fn, gen_optimizer,
                                                          fake_data, real_labels)

            # Save Losses for plotting later
            disc_loss.append(disc_error)
            gen_loss.append(gen_error)

            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(data_loader)}]\n'
                      f'Loss_D: {disc_error:.4f}\n'
                      f'Loss_G: {gen_error:.4f}\n'
                      f'D(x): {dr_o_x:.4f}\tD(G(z)): {dr_g_o_z1:.4f} / {dr_g_o_z2:.4f}')

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(data_loader) - 1)):
                with torch.no_grad():
                    fake_images = generator_net(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

            iters += 1

    return generator_net, discriminator_net, gen_loss, disc_loss, img_list


def main():
    # Read command line args
    config_path, load_models, debug_enabled, plot_enabled = process_args(load_args())
    # Load config file
    conf = load_config_file(config_path)

    # Environment conf
    seed_min = int(conf['environment']['seed_min'])
    seed_max = int(conf['environment']['seed_max'])
    n_gpu = int(conf['environment']['n_gpu'])
    # Data conf
    batch_size = int(conf['data']['batch_size'])
    dl_workers = int(conf['data']['dl_workers'])
    data_root = conf['data']['data_root']
    image_size = int(conf['data']['image_size'])
    img_channels = int(conf['data']['img_channels'])
    norm = float(conf['data']['norm'])
    shuffle = True  # Todo parse to boolean
    # Training conf
    lr = float(conf['training']['lr'])
    num_epochs = int(conf['training']['num_epochs'])
    beta1 = float(conf['training']['beta1'])
    # Network conf
    noise_vector_size = int(conf['net']['noise_vector_size'])
    g_fm_depth = int(conf['net']['g_fm_depth'])
    d_fm_depth = int(conf['net']['d_fm_depth'])
    gen_model_path = ['net']['gen_model_path']
    disc_model_path = ['net']['disc_model_path']

    # Sets the device (GPU or CPU) and creates the data_loader
    device = set_environment((seed_min, seed_max), n_gpu)
    data_loader = load_data(data_root=data_root, img_size=image_size, batch_size=batch_size,
                            norm=norm, shuffle=shuffle, data_loader_workers=dl_workers)

    # Create or load Generator and Discriminator models
    if load_models:
        generator_net = Generator(n_gpu, noise_vector_size, fm_depth=g_fm_depth, img_nc=img_channels)
        generator_net.load_state_dict(torch.load(gen_model_path))
        generator_net.eval()
        discriminator_net = Discriminator(n_gpu=n_gpu, fm_depth=d_fm_depth, img_nc=img_channels)
        discriminator_net.load_state_dict(torch.load(disc_model_path))
        discriminator_net.eval()
    else:
        generator_net = Generator(n_gpu, noise_vector_size, fm_depth=g_fm_depth, img_nc=img_channels)
        discriminator_net = Discriminator(n_gpu=n_gpu, fm_depth=d_fm_depth, img_nc=img_channels)
    # Create loss function(BCE) and both model optimizers
    loss_fn, gen_optimizer, disc_optimizer = create_bce_loss_optimizers(generator_net,
                                                                        discriminator_net,
                                                                        learning_rate=lr,
                                                                        beta=beta1)
    # Executes training loop
    generator_net, discriminator_net, gen_loss, disc_loss, img_list = train(device, data_loader,
                                                                            generator_net,
                                                                            discriminator_net, loss_fn,
                                                                            disc_optimizer, gen_optimizer,
                                                                            num_epochs=num_epochs,
                                                                            noise_v_size=noise_vector_size)
    # Plot training results
    plot_loss_results(disc_loss, gen_loss)
    show_generated_images(8, 8, img_list)
    # Save models
    torch.save(generator_net.state_dict(), gen_model_path)
    torch.save(discriminator_net.state_dict(), disc_model_path)


if __name__ == "__main__":
    main()
