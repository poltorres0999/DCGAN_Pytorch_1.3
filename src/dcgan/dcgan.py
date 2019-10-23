from __future__ import print_function

import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dataset_utils
import torchvision.transforms as transforms
import torchvision.utils as vutils
from IPython.display import HTML
from datetime import datetime

from src.dcgan.utils import save_model

DEFAULT_CONFIG_PATH = "../../res/default_config.ini"


class DCGAN():

    def __init__(self, generator, discriminator, device, data_loader):
        self.generator_net = generator
        self.discriminator_net = discriminator
        self.device = device
        self.data_loader = data_loader
        self.gen_optimizer = None
        self.disc_optimizer = None
        self.loss_fn = None
        self.gen_loss = []
        self.disc_loss = []
        self.generated_images = []

    def plot_images(self, img_data, num_images, fig_size, device, plt_title=""):
        plt.figure(figsize=(fig_size, fig_size))
        plt.axis("off")
        plt.title(plt_title)
        plt.imshow(np.transpose(
            vutils.make_grid(img_data.to(device)[:num_images], padding=2, normalize=True).cpu(), (1, 2, 0)))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def show_generated_images(self, num_images_x, num_images_y):
        fig = plt.figure(figsize=(num_images_x, num_images_y))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in self.generated_images]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_optimizers(self, disc_optimizer, gen_optimizer):
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer

    # todo: enable choosing different optimizer and loss function (split in two functions)
    def set_bce_loss_optimizers(self, learning_rate, beta):
        # Initialize BCELoss function
        self.loss_fn = nn.BCELoss()
        # Setup Adam optimizers for both G and D
        self.disc_optimizer = optim.Adam(self.discriminator_net.parameters(), lr=learning_rate, betas=(beta, 0.999))
        self.gen_optimizer = optim.Adam(self.generator_net.parameters(), lr=learning_rate, betas=(beta, 0.999))

    def optimize_discriminator_net(self, real_data, fake_data,
                                   real_labels, fake_labels):
        ############################
        # Update Discriminator network: maximize log(D(G(z)))
        ###########################
        # -------- Train discriminator with all-real batch ------------
        # Set discriminator gradient values to 0
        self.discriminator_net.zero_grad()
        # Forward pass real batch through D
        # view(-1) -> Automatically re-shapes
        output = self.discriminator_net(real_data).view(-1)
        # Calculate loss on all-real batch
        disc_err_real = self.loss_fn(output, real_labels)
        # Calculate gradients for D in backward pass
        disc_err_real.backward()
        D_x = output.mean().item()

        # ------- Train with all-fake batch -------
        # Classify all fake batch with Discriminator
        output = self.discriminator_net(fake_data.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        disc_err_fake = self.loss_fn(output, fake_labels)
        # Calculate the gradients for this batch
        disc_err_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        disc_error = disc_err_real + disc_err_fake
        # Update discriminator with the calculated gradients
        self.disc_optimizer.step()

        return D_x, D_G_z1, disc_error

    def optimize_generator_net(self, fake_data, labels):
        ############################
        # Update Generator network: maximize log(D(G(z)))
        ###########################
        self.generator_net.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.discriminator_net(fake_data).view(-1)
        # Calculate Generator's loss based on Discriminator's output
        gen_error = self.loss_fn(output, labels)
        # Calculate gradients for Generator
        gen_error.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.gen_optimizer.step()

        return gen_error, D_G_z2

    # todo: Debug
    # todo: store images as png
    def train(self, gen_save_path, disc_save_path, images_save_path ,num_epochs=1, noise_v_size=100, real_label_v=1, fake_label_v=0, store_frequency=500,
              debug=False):
        # Training Loop
        # Create batch of latent vectors that we will use to visualize the progression of the generator
        fixed_noise = torch.randn(64, noise_v_size, 1, 1, device=self.device)
        iters = 0
        current_epoch = 0
        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            print(f"Started epoch [{epoch}] / [{num_epochs}]")
            # For each batch in the data_loader
            # Note -> enumerate outputs (int index, item from iterable object)
            current_epoch = epoch
            for i, train_data in enumerate(self.data_loader):
                print(f"Iteration [{i}] / [{len(self.data_loader)}]")
                # ---Prepare data for the current training step---
                real_data = train_data[0].to(self.device)
                # Get batch size
                batch_size = real_data.size(0)
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, noise_v_size, 1, 1, device=self.device)
                # Generate fake image batch with Generator
                fake_data = self.generator_net(noise)
                # Create labels
                real_labels = torch.full((batch_size,), real_label_v, device=self.device)
                fake_labels = torch.full((batch_size,), fake_label_v, device=self.device)

                disc_error, dr_o_x, dr_g_o_z1 = self.optimize_discriminator_net(real_data, fake_data, real_labels,
                                                                                fake_labels)
                # fake labels are real for generator cost (maximize log(D(G(z))))
                gen_error, dr_g_o_z2 = self.optimize_generator_net(fake_data, real_labels)

                # Save Losses for plotting later
                self.disc_loss.append(disc_error)
                self.gen_loss.append(gen_error)

                # Output training stats
                if i % 50 == 0:
                    print(f'[{epoch}/{num_epochs}][{i}/{len(self.data_loader)}]\n'
                          f'Loss_D: {disc_error:.4f}\n'
                          f'Loss_G: {gen_error:.4f}\n'
                          f'D(x): {dr_o_x:.4f}\tD(G(z)): {dr_g_o_z1:.4f} / {dr_g_o_z2:.4f}')

                # Check how the generator is doing by saving G's output on fixed_noise
                if (store_frequency % 240 == 0) or ((epoch == num_epochs - 1) and (i == len(self.data_loader) - 1)):
                    with torch.no_grad():
                        fake_images = self.generator_net(fixed_noise).detach().cpu()
                        fake_images_path = f"{images_save_path}/{datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}.png"
                        vutils.save_image(fake_images, fake_images_path)
                    self.generated_images.append(vutils.make_grid(fake_images, padding=2, normalize=True))

                iters += 1

            save_model(self.generator_net, self.gen_optimizer, epoch, gen_save_path)
            save_model(self.discriminator_net, self.disc_optimizer, epoch, disc_save_path)
