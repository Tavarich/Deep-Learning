import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import UNet, DeepLab


def predict(model, path, save):
    if not os.path.exists(save):
        os.makedirs(save)


    image = Image.open(path)
    w, h = image.size
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)

    image.unsqueeze_(0)
    image = image.to(device)
    pred = model(image).cpu()  # ['out'].cpu()
    pred = pred.argmax(dim=1, keepdim=True).float()
    pred.squeeze_(0)
    mask = transforms.ToPILImage()(pred)
    mask = transforms.Resize((h, w))(mask)
    mask = np.asarray(mask)

    name = 'pred.png'
    cv2.imwrite(os.path.join(save, name), mask)



if __name__ == '__main__':
    unet = UNet()
    # unet = DeepLab()
    device = torch.device('cuda')
    unet.to(device)
    unet.eval()
    checkpoint = torch.load('/root/autodl-tmp/Models/MedicalSegment/unet.pth')
    unet.load_state_dict(checkpoint['model'])
    path = '/root/autodl-tmp/Blood/test/img'
    save = '/root/autodl-tmp/Blood/test/pred'

    predict(unet, path, save)


