import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力模块
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ResNet的BasicBlock残差块
class BasicBlock(nn.Module):
    expansion = 1  # 通道升降维倍数

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1)  # 第一个卷积层，通过stride进行下采样
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,  # 第二个卷积层，不进行下采样
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.ca = ChannelAttention(channels)  # 加入卷积注意力模块
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)  # 1x1卷积层用于降采样

        out += residual
        return self.relu(out)


# ResNet的Bottleneck残差块
class Bottleneck(nn.Module):
    expansion = 4  # 通道升降维背书

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels,  # 第一个是1x1卷积
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,  # 第二个是3x3卷积，通过stride进行下采样
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels * self.expansion,  # 第三个是1x1卷积
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * channels)

        self.ca = ChannelAttention(channels * self.expansion)  # 加入卷积注意力模块
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample:
            residual = self.downsample(x)  # 通道数变化，需要在shortcut中加入1x1卷积升维，同时降采样

        out += residual

        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=200):
        self.in_channels = 64
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])  # 第一个残差层不进行下采样
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, 1000)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 200)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, blocks, stride=1):
        """
        :param block: 残差块类别，BiascBlock or Bottleneck
        :param channels: 通道数
        :param blocks: 残差块数量
        :param stride: 卷积核步长
        :return: nn.Sequential
        """
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:  # 前一种操作需要下采样，后一种操作需要融合通道
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))  # 第一个残差块
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):  # 后续残差块，需要改变in_channels，使其对应上一个残差块的channels
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)

        return x


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet34_cbam(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


if __name__ == "__main__":
    model = resnet34_cbam(True)
    print(model)
