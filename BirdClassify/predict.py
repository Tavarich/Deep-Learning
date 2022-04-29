import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import Counter

from model import resnet34_cbam
from datasets import get_image_list

from datasets import match_dict


def predict(models, test_list, image_folder, pred_path, map):
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    with open(pred_path + "pred.csv", "w") as f:
        for i in range(len(test_list)):
            name = str(i + 1) + ".jpg"
            ipath = os.path.join(image_folder, name)
            img = Image.open(ipath).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(dim=0)
            img = img.to(device)
            vote = []
            for mlp in models:
                pred = mlp(img).argmax(dim=1)
                vote.append(pred.cpu().numpy())
            vote = np.array(vote)
            label = Counter(vote[:, 0]).most_common(1)[0][0] + 1
            f.write("{},{}\n".format(name, label))
            if i % 100 == 0:
                print(name, map[label])


if __name__ == "__main__":
    test_path = "classify-birds/test_set"
    test_list = get_image_list(test_path)
    pred_path = ""

    device = torch.device('cuda')
    models = [resnet34_cbam() for i in range(5)]

    checkpoint = torch.load('checkpoint')

    for i, model in enumerate(models):
        model.to(device)
        model.load_state_dict(checkpoint['model'][i])
        model.eval()

    predict(models, test_list, test_path, pred_path, match_dict)
