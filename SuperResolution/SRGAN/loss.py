import torch
import torch.nn as nn
from torchvision.models import vgg19


class ContentLoss(nn.Module):
    # 内容损失
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss()
        self.vgg = nn.Sequential(*list(vgg19(pretrained=True).features[:34]))
        self.vgg = self.vgg.to(device)

    def forward(self, fake, real):
        feature_fake = self.vgg(fake)
        feature_real = self.vgg(real)
        loss = self.mse(feature_fake, feature_real)
        return loss


class AdversarialLoss(nn.Module):
    # 对抗损失
    def __init__(self):
        super().__init__()

    def forward(self, x):
        loss = torch.sum(-torch.log(x))
        return loss


class PerceptualLoss(nn.Module):
    # 感知损失，包括内容损失和对抗损失
    def __init__(self, device):
        super().__init__()
        self.vgg_loss = ContentLoss(device)
        self.adversarial_loss = AdversarialLoss()

    def forward(self, fake, real, x):
        vgg_loss = self.vgg_loss(fake, real)
        adversarial_loss = self.adversarial_loss(x)

        return vgg_loss + 1e-3 * adversarial_loss
    

if __name__ == "__main__":
    print(nn.Sequential(*list(vgg19().features)[:34]))
    
    print(vgg19())
