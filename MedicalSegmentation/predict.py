import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import UNet, DeepLab, UNetPP, PAN, PSPNet


def predict(model, path, save):
    if not os.path.exists(save):
        os.makedirs(save)

    for img in os.listdir(path):
        if img == '.ipynb_checkpoints':
            continue
        image = Image.open(os.path.join(path, img))
        w, h = image.size
        image = transforms.Resize((512, 512))(image)
        image = transforms.ToTensor()(image)
       
        image.unsqueeze_(0)
        image = image.to(device)
        pred = model(image).cpu()#['out'].cpu()     
        pred = pred.argmax(dim=1, keepdim=True).float()
        pred.squeeze_(0)
        mask = transforms.ToPILImage()(pred)
        mask = transforms.Resize((h, w))(mask)
        mask = np.asarray(mask)
        
        cv2.imwrite(os.path.join(save, img), mask)
        
        print(f'{img} done')


if __name__ == '__main__':
    #unet = UNet()
    #unet = DeepLab()
    
    x = input("1. unet or 2. unetpp?(1/2)")
    if x == '1':
        path = '/root/autodl-tmp/Models/MedicalSegment/unet.pth'
        net = UNet()
    elif x == '2':
        path = '/root/autodl-tmp/Models/MedicalSegment/unetpp.pth'
        net = UNetPP()
    device = torch.device('cuda')
    net.to(device)
    net.eval()
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model'])
    path = '/root/autodl-tmp/Blood/test/img'
    save = '/root/autodl-tmp/Blood/test/pred'

    predict(net, path, save)
