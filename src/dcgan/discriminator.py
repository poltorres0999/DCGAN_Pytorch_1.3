from __future__ import print_function
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_gpu, fm_depth, img_nc):
        super(Discriminator, self).__init__()
        self.ngpu = n_gpu
        self.fm_depth = fm_depth
        self.img_nc = img_nc
        self.main = nn.Sequential(
            # input is (image_nc) x 64 x 64
            nn.Conv2d(self.img_nc, self.fm_depth, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            # output depth -> fm_depth = 64
            nn.Conv2d(self.fm_depth, self.fm_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            # output depth -> fm_depth * 2= 128
            nn.Conv2d(self.fm_depth * 2, self.fm_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # output depth -> fm_depth * 4 = 256
            nn.Conv2d(self.fm_depth * 4, self.fm_depth * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # output depth -> fm_depth = 512
            nn.Conv2d(self.fm_depth * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# SevGan uses image size of 256x256, kernel size = 4 x 4, stride=2, noise_size = 128
class SevDiscriminator(nn.Module):
    def __init__(self, n_gpu, fm_depth, img_nc):
        super(SevDiscriminator, self).__init__()
        self.ngpu = n_gpu
        self.fm_depth = fm_depth
        self.img_nc = img_nc
        self.main = nn.Sequential(
            # input is 3 x 256 x 256
            nn.Conv2d(self.img_nc, int(self.fm_depth / 2), 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(int(self.fm_depth / 2), self.fm_depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(self.fm_depth, self.fm_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(self.fm_depth * 2, self.fm_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(self.fm_depth * 4, self.fm_depth * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.fm_depth * 8, self.fm_depth * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.fm_depth * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# RasGAN uses image size of 256x256, kernel size = 4 x 4, stride=2, noise_size = 128
class SixDiscriminator(nn.Module):
    def __init__(self, n_gpu, fm_depth, img_nc):
        super(SixDiscriminator, self).__init__()
        self.ngpu = n_gpu
        self.fm_depth = fm_depth
        self.img_nc = img_nc
        self.main = nn.Sequential(
            # input is 3 x 128 x 128
            nn.Conv2d(self.img_nc, self.fm_depth, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(self.fm_depth, self.fm_depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(self.fm_depth * 2, self.fm_depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.fm_depth * 4, self.fm_depth * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.fm_depth * 8, self.fm_depth * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.fm_depth * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class SmallKernelDiscriminator(nn.Module):
    def __init__(self, n_gpu, fm_depth, img_nc):
        super(SmallKernelDiscriminator, self).__init__()
        self.ngpu = n_gpu
        self.fm_depth = fm_depth
        self.img_nc = img_nc
        self.main = nn.Sequential(
            # input is (image_nc) x 64 x 64
            nn.Conv2d(self.img_nc, self.fm_depth, 3, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            # output depth -> fm_depth = 64
            nn.Conv2d(self.fm_depth, self.fm_depth * 2, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            # output depth -> fm_depth * 2= 128
            nn.Conv2d(self.fm_depth * 2, self.fm_depth * 4, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # output depth -> fm_depth * 4 = 256
            nn.Conv2d(self.fm_depth * 4, self.fm_depth * 8, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # output depth -> fm_depth = 512
            nn.Conv2d(self.fm_depth * 8, 1, 3, 3, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class SmallKernelSixDiscriminator(nn.Module):
    def __init__(self, n_gpu, fm_depth, img_nc):
        super(SmallKernelSixDiscriminator, self).__init__()
        self.ngpu = n_gpu
        self.fm_depth = fm_depth
        self.img_nc = img_nc
        self.main = nn.Sequential(
            # input is (image_nc) x 64 x 64
            nn.Conv2d(self.img_nc, self.fm_depth, 3, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            # output depth -> fm_depth = 64
            nn.Conv2d(self.fm_depth, self.fm_depth * 2, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            # output depth -> fm_depth * 2= 128
            nn.Conv2d(self.fm_depth * 2, self.fm_depth * 3, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 3),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            # output depth -> fm_depth * 2= 128
            nn.Conv2d(self.fm_depth * 3, self.fm_depth * 4, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # output depth -> fm_depth * 4 = 256
            nn.Conv2d(self.fm_depth * 4, self.fm_depth * 5, 3, 3, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 5),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # output depth -> fm_depth = 512
            nn.Conv2d(self.fm_depth * 5, 1, 3, 3, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class FiveKernelDiscriminator(nn.Module):
    def __init__(self, n_gpu, fm_depth, img_nc):
        super(FiveKernelDiscriminator, self).__init__()
        self.ngpu = n_gpu
        self.fm_depth = fm_depth
        self.img_nc = img_nc
        self.main = nn.Sequential(
            # input is (image_nc) x 64 x 64
            nn.Conv2d(self.img_nc, self.fm_depth, 5, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            # output depth -> fm_depth = 64
            nn.Conv2d(self.fm_depth, self.fm_depth * 2, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            # output depth -> fm_depth * 2= 128
            nn.Conv2d(self.fm_depth * 2, self.fm_depth * 4, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # output depth -> fm_depth * 4 = 256
            nn.Conv2d(self.fm_depth * 4, self.fm_depth * 8, 5, 2, 1, bias=False),
            nn.BatchNorm2d(self.fm_depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # output depth -> fm_depth = 512
            nn.Conv2d(self.fm_depth * 8, 1, 5, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
