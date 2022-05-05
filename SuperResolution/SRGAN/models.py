import torch
import torch.nn as nn
from math import log2

class Residual(nn.Module):
    def __init__(self, kernel_size=3, channels=64, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.prelu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += x
        return y


def make_res_layer(num_blocks=16):
    layer = []
    for i in range(num_blocks):
        layer.append(Residual())
    return nn.Sequential(*layer)


class UpSample(nn.Module):
    def __init__(self, channels, scaling_size):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scaling_size ** 2, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(scaling_size)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.prelu(x)

        return x


def make_upsample_layer(num_feature, scaling_size):
    assert scaling_size in [2, 3, 4, 8]
    num_upsample = int(log2(scaling_size))
    layer = []
    for i in range(num_upsample):
        if scaling_size == 3:
            layer.append(UpSample(num_feature, scaling_size=3))
        else:
            layer.append(UpSample(num_feature, scaling_size=2))
    return nn.Sequential(*layer)

class Generator(nn.Module):
    in_channels = 3
    channels = 64

    def __init__(self, scaling_size):
        super().__init__()
        self.conv1 = nn.Conv2d(self.in_channels, self.channels, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()

        self.layer = make_res_layer()

        self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(self.channels)

        self.upsample = make_upsample_layer(self.channels, scaling_size)

        self.conv3 = nn.Conv2d(self.channels, self.in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)

        y = self.layer(x)
        y = self.conv2(y)
        y = self.bn(y)

        y += x

        y = self.upsample(y)
        y = self.conv3(y)

        return y


class VggBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


def make_vgg_layer(cfg):
    layer = []
    for channel in cfg:
        in_ = channel[0]
        out = channel[1]
        stride = 2 // (out // in_)

        layer.append(VggBlock(in_, out, stride))

    return nn.Sequential(*layer)


class Discriminator(nn.Module):
    in_channels = 3
    channels = 64
    cfg = [(64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 512), (512, 512)]

    def __init__(self, crop_size):
        super().__init__()

        self.size = crop_size // 16

        self.conv = nn.Conv2d(self.in_channels, self.channels, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.layer = make_vgg_layer(self.cfg)
        
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
       
        x = self.conv(x)
        x = self.lrelu(x)
        x = self.layer(x)
        x = self.dense(x)
        
        return x


if __name__ == "__main__":
    g = Generator(scaling_size=2)
    d = Discriminator(crop_size=96)
    x = torch.randn(3, 3, 96, 96)
    print(g(x).shape)
    print(d(x).shape)
    