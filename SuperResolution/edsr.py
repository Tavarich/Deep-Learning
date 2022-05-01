import torch
import torch.nn as nn
from math import log2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        """
        :param channels:    输入通道数
        :param kernel_size: 卷积核大小
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y += x * 0.1
        return y


class UpSample(nn.Module):
    def __init__(self, channels, scaling_size=2):
        """
        :param channels:      输入通道数
        :param scaling_size:  放大倍率
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, scaling_size ** 2 * channels, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scaling_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x


def make_layer(num_layer, channels):
    """
    构建网络中的残差块层
    :param num_layer: 残差块数量
    :param channels:  输入通道数
    :return: nn.Sequential
    """
    layer = []
    for i in range(num_layer):
        layer.append(ResBlock(channels, kernel_size=3))
    return nn.Sequential(*layer)

def make_upsample_layer(num_feature, scaling_size):
    """
    构建网络中上采样层
    :param num_feature: 输入通道数
    :param scaling_size: 放大倍率
    :return: nn.Sequential
    """
    assert scaling_size in [2, 3, 4, 8]
    num_upsample = int(log2(scaling_size))
    layer = []
    for i in range(num_upsample):
        if scaling_size == 3:
            layer.append(UpSample(num_feature, scaling_size=3))
        else:
            layer.append(UpSample(num_feature, scaling_size=2))
    return nn.Sequential(*layer)


class EDSR(nn.Module):
    def __init__(self, num_resblock=32, num_feature=256, scaling_size=2):
        super().__init__()
        self.scaling_size = scaling_size
        self.num_layer = num_resblock
        
        self.conv1 = nn.Conv2d(3, num_feature, kernel_size=3, padding=1)
        
        self.resblocks = make_layer(self.num_layer, num_feature)
        
        self.upsample = make_upsample_layer(num_feature, scaling_size)
        
        self.conv2 = nn.Conv2d(num_feature, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        y = self.resblocks(x)
        y += 0.1 * x
        y = self.upsample(y)
        y = self.conv2(y)
        return y


if __name__ == "__main__":
    edsr = EDSR()
    x = torch.randn((1, 3, 224, 224))
    print(edsr)
    print(edsr(x))