import torch
from torch import optim
from torchvision import utils as vutils
from src.dcgan.utils import create_GAN_model, load_model, generate_fake_images, generate_images_from_single_noise

model_path = "../../results/CASIA_results/SmallKernel_23-11-2019_07-31-52_PM/models/generator.pt"
store_path = "C:/Users/polto/Desktop/SmallKernelImages"
model_name = "SmallKernel"
img_channels = 3
n_gpu = 1
noise_vector_size = 100
fm_depth = 163
lr = 0.0002
beta1=0.5
device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")
alpha = 0.0001
n_samples = 2000
generator, discriminator = create_GAN_model(model_name, n_gpu, noise_vector_size, fm_depth, img_channels)
gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
model = load_model(generator, gen_optimizer, model_path)
generator.cuda()
generate_fake_images(batch_size=128, noise_v_size=noise_vector_size, generator=generator, device=device, images_save_path=store_path)

"""
generate_images_from_single_noise(generator=generator, device=device, images_save_path=store_path,
                                 noise_v_size=noise_vector_size, n_samples=n_samples, alpha=alpha)
"""



