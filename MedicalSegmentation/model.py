import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        down = self.downsample(out)
        return out, down


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channel * 2, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, out):
        x = self.conv(x)
        up = self.upsample(x)
        cat_out = torch.cat([up, out], dim=1)
        return cat_out


class UNet(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.encoder1 = Encoder(3, 64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)

        self.decoder4 = Decoder(512, 512)  # 512, 512
        self.decoder3 = Decoder(1024, 256)
        self.decoder2 = Decoder(512, 128)
        self.decoder1 = Decoder(256, 64)

        self.output = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_class, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out1, down1 = self.encoder1(x)
        out2, down2 = self.encoder2(down1)
        out3, down3 = self.encoder3(down2)
        out4, down4 = self.encoder4(down3)

        up4 = self.decoder4(down4, out4)
        up3 = self.decoder3(up4, out3)
        up2 = self.decoder2(up3, out2)
        up1 = self.decoder1(up2, out1)

        out = self.output(up1)
        return out


class DeepLab(nn.Module):
    def __init__(self, pretrained=False, num_classes=2):
        super().__init__()
        self.net = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        return self.net(x)['out']
    

class UNetPP(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )

    def forward(self, x):
        out = self.model(x)
        return out


class PAN(nn.Module):
    def __init__(self, num_classes=2):
        super(PAN, self).__init__()
        self.model = smp.PAN(
            in_channels=3,
            classes=num_classes
        )

    def forward(self, x):
        out = self.model(x)
        return out

class PSPNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PSPNet, self).__init__()
        self.model = smp.PSPNet(
            in_channels=3,
            classes=num_classes
        )

    def forward(self, x):
        out = self.model(x)
        return out

if __name__ == '__main__':
    unet = smp.Unet(classes=2)
    x = torch.randn(10, 3, 256, 256)
    # y = unet(x)
    # print(y.shape)
    # unetpp = UNetPP()
    # print(unetpp(x).shape)
    print(unet(x).shape)
