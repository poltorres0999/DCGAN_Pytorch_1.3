from __future__ import print_function
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, n_gpu=0, noise_vector_size=1, fm_depth=1, img_nc=1):
        super(Generator, self).__init__()
        self.ngpu = n_gpu
        self.noise_vector_size = noise_vector_size
        self.fm_depth = fm_depth
        self.img_nc = img_nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector_size, self.fm_depth * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.ReLU(True),
            # state size. (fm_depth * 8) x 4 x 4
            # output depth -> 64 *  8 = 512
            nn.ConvTranspose2d(self.fm_depth * 8, self.fm_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.ReLU(True),
            # state size. (fm_depth*4) x 8 x 8
            # output depth -> 64 *  4 = 256
            nn.ConvTranspose2d(self.fm_depth * 4, self.fm_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.ReLU(True),
            # state size. (fm_depth*2) x 16 x 16
            # output depth -> 64 *  2 = 128
            nn.ConvTranspose2d(self.fm_depth * 2, self.fm_depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.ReLU(True),
            # state size. (fm_depth) x 32 x 32
            # output depth -> 64 *  8 = 64
            nn.ConvTranspose2d(self.fm_depth, img_nc, 4, 2, 1, bias=False),
            nn.Tanh())
            # state size. (img_nc) x 64 x 64
            # output depth -> img_nc = 3

    def forward(self, input_data):
        return self.main(input_data)


# RasGAN uses image size of 256x256, kernel size = 4 x 4, stride=2, noise_size = 128
class SevGenerator(nn.Module):
    def __init__(self, n_gpu=0, noise_vector_size=128, fm_depth=256, img_nc=3):
        super(SevGenerator, self).__init__()
        self.ngpu = n_gpu
        self.noise_vector_size = noise_vector_size
        self.fm_depth = fm_depth
        self.img_nc = img_nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector_size, self.fm_depth * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 16),
            nn.ReLU(True),
            # state size. 4096 x 4 x 4
            nn.ConvTranspose2d(self.fm_depth * 16, self.fm_depth * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.ReLU(True),
            # state size. 2048 x 8 x 8
            nn.ConvTranspose2d(self.fm_depth * 8, self.fm_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.ReLU(True),
            # state size. 1024 x 16 x 16
            nn.ConvTranspose2d(self.fm_depth * 4, self.fm_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.ReLU(True),
            # state size. 512 x 32 x 32
            nn.ConvTranspose2d(self.fm_depth * 2, self.fm_depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.ReLU(True),
            # state size. 256 x 64 x 64
            nn.ConvTranspose2d(self.fm_depth, int(self.fm_depth/2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(self.fm_depth / 2)),
            nn.ReLU(True),
            # state size. 128 x 128 x 128
            nn.ConvTranspose2d(int(self.fm_depth / 2), img_nc, 4, 2, 1, bias=False),
            nn.Tanh())
            # state size. 3 x 256 x 256

    def forward(self, input_data):
        return self.main(input_data)


class SixGenerator(nn.Module):
    def __init__(self, n_gpu=0, noise_vector_size=100, fm_depth=128, img_nc=3):
        super(SixGenerator, self).__init__()
        self.ngpu = n_gpu
        self.noise_vector_size = noise_vector_size
        self.fm_depth = fm_depth
        self.img_nc = img_nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector_size, self.fm_depth * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 16),
            nn.ReLU(True),
            # state size. 2048 x 4 x 4
            nn.ConvTranspose2d(self.fm_depth * 16, self.fm_depth * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.ReLU(True),
            # state size. 1024 x 8 x 8
            nn.ConvTranspose2d(self.fm_depth * 8, self.fm_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.ReLU(True),
            # state size. 512 x 16 x 16
            nn.ConvTranspose2d(self.fm_depth * 4, self.fm_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.ReLU(True),
            # state size. 256 x 32 x 32
            nn.ConvTranspose2d(self.fm_depth * 2, self.fm_depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.ReLU(True),
            # state size. 128 x 64 x 64
            nn.ConvTranspose2d(self.fm_depth, img_nc, 4, 2, 1, bias=False),
            nn.Tanh())
        # state size. 3 x 128 x 128
    def forward(self, input_data):
            return self.main(input_data)


class SmallKernelGenerator(nn.Module):
    def __init__(self, n_gpu=0, noise_vector_size=1, fm_depth=1, img_nc=1):
        super(SmallKernelGenerator, self).__init__()
        self.ngpu = n_gpu
        self.noise_vector_size = noise_vector_size
        self.fm_depth = fm_depth
        self.img_nc = img_nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector_size, self.fm_depth * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.ReLU(True),
            # state size. (fm_depth * 8) x 3 x 3
            # output depth -> 64 *  8 = 512
            nn.ConvTranspose2d(self.fm_depth * 8, self.fm_depth * 4, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.ReLU(True),
            # state size. (fm_depth*4) x 7 x 7
            # output depth -> 64 *  4 = 256
            nn.ConvTranspose2d(self.fm_depth * 4, self.fm_depth * 2, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.ReLU(True),
            # state size. (fm_depth*2) x 19 x 19
            # output depth -> 64 *  2 = 128
            nn.ConvTranspose2d(self.fm_depth * 2, self.fm_depth, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.ReLU(True),
            # state size. (fm_depth) x 55 x 55
            # output depth -> 64 *  8 = 64
            nn.ConvTranspose2d(self.fm_depth, img_nc, 3, 3, 1, bias=False),
            nn.Tanh())
            # state size. (img_nc) x 163 x 163
            # output depth -> img_nc = 3

    def forward(self, input_data):
        return self.main(input_data)


class SmallKernelSixGenerator(nn.Module):
    def __init__(self, n_gpu=0, noise_vector_size=1, fm_depth=1, img_nc=1):
        super(SmallKernelSixGenerator, self).__init__()
        self.ngpu = n_gpu
        self.noise_vector_size = noise_vector_size
        self.fm_depth = fm_depth
        self.img_nc = img_nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector_size, self.fm_depth * 5, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 5),
            nn.ReLU(True),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.fm_depth * 5, self.fm_depth * 4, 3, 3, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.ReLU(True),
            # state size. (fm_depth * 8) x 3 x 3
            # output depth -> 64 *  8 = 512
            nn.ConvTranspose2d(self.fm_depth * 4, self.fm_depth * 3, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 3),
            nn.ReLU(True),
            # state size. (fm_depth*4) x 7 x 7
            # output depth -> 64 *  4 = 256
            nn.ConvTranspose2d(self.fm_depth * 3, self.fm_depth * 2, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.ReLU(True),
            # state size. (fm_depth*2) x 19 x 19
            # output depth -> 64 *  2 = 128
            nn.ConvTranspose2d(self.fm_depth * 2, self.fm_depth, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.ReLU(True),
            # state size. (fm_depth) x 55 x 55
            # output depth -> 64 *  8 = 64
            nn.ConvTranspose2d(self.fm_depth, img_nc, 3, 3, 1, bias=False),
            nn.Tanh())
            # state size. (img_nc) x 163 x 163
            # output depth -> img_nc = 3

    def forward(self, input_data):
        return self.main(input_data)


class FiveKernelGenerator(nn.Module):
    def __init__(self, n_gpu=0, noise_vector_size=1, fm_depth=1, img_nc=1):
        super(FiveKernelGenerator, self).__init__()
        self.ngpu = n_gpu
        self.noise_vector_size = noise_vector_size
        self.fm_depth = fm_depth
        self.img_nc = img_nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector_size, self.fm_depth * 8, 5, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.ReLU(True),
            # state size. (fm_depth * 8) x 4 x 4
            # output depth -> 64 *  8 = 512
            nn.ConvTranspose2d(self.fm_depth * 8, self.fm_depth * 4, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.ReLU(True),
            # state size. (fm_depth*4) x 8 x 8
            # output depth -> 64 *  4 = 256
            nn.ConvTranspose2d(self.fm_depth * 4, self.fm_depth * 2, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.ReLU(True),
            # state size. (fm_depth*2) x 16 x 16
            # output depth -> 64 *  2 = 128
            nn.ConvTranspose2d(self.fm_depth * 2, self.fm_depth, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.ReLU(True),
            # state size. (fm_depth) x 32 x 32
            # output depth -> 64 *  8 = 64
            nn.ConvTranspose2d(self.fm_depth, img_nc, 5, 2, 1, bias=False),
            nn.Tanh())
            # state size. (img_nc) x 64 x 64
            # output depth -> img_nc = 3

    def forward(self, input_data):
        return self.main(input_data)


class FiveKernelSixLayerGenerator(nn.Module):
    def __init__(self, n_gpu=0, noise_vector_size=1, fm_depth=1, img_nc=1):
        super(FiveKernelSixLayerGenerator, self).__init__()
        self.ngpu = n_gpu
        self.noise_vector_size = noise_vector_size
        self.fm_depth = fm_depth
        self.img_nc = img_nc

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.noise_vector_size, self.fm_depth * 12 , 5, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 12),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.fm_depth * 12, self.fm_depth * 8, 5, 1, 0, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.ReLU(True),
            # state size. (fm_depth * 8) x 4 x 4
            # output depth -> 64 *  8 = 512
            nn.ConvTranspose2d(self.fm_depth * 8, self.fm_depth * 4, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.ReLU(True),
            # state size. (fm_depth*4) x 8 x 8
            # output depth -> 64 *  4 = 256
            nn.ConvTranspose2d(self.fm_depth * 4, self.fm_depth * 2, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.ReLU(True),
            # state size. (fm_depth*2) x 16 x 16
            # output depth -> 64 *  2 = 128
            nn.ConvTranspose2d(self.fm_depth * 2, self.fm_depth, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.ReLU(True),
            # state size. (fm_depth) x 32 x 32
            # output depth -> 64 *  8 = 64
            nn.ConvTranspose2d(self.fm_depth, img_nc, 5, 2, 1, bias=False),
            nn.Tanh())
            # state size. (img_nc) x 64 x 64
            # output depth -> img_nc = 3

    def forward(self, input_data):
        return self.main(input_data)