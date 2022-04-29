import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from model import resnet34_cbam

from datasets import match_dict

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def predit_img(model, image):
    image = transform(image)
    image = image.unsqueeze(dim=0)
    pred = model(image).argmax(dim=1).item() + 1
    return match_dict[pred]

if __name__ == '__main__':
    image_path = "classify-birds/train_set/001.Black_footed_Albatross_283.jpg"
    image = Image.open(image_path)
    
    model = resnet34_cbam()
    
    checkpoint = torch.load('checkpoint')
    
    model.load_state_dict(checkpoint['model'][0])
    model.eval()
    bird = predit_img(model, image)
    print(f'this bird is {bird}')
