import os
import random
import numpy as np
import matplotlib.pyplot as plt
import elasticdeform
from torchvision.transforms import transforms, functional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from PIL import Image

def get_kfold(img_dir, k):

    kf = KFold(n_splits=k, shuffle=True)
    lst = [i + 1 for i in range(len(os.listdir(img_dir)))]
    train_list = []
    valid_list = []
    for x, y in kf.split(lst):
        # x是训练集索引，y是验证集索引
        x = [str(i + 1) + '.png' for i in x]
        y = [str(i + 1) + '.png' for i in y]
        train_list.append(x)
        valid_list.append(y)
    return train_list, valid_list


def load_images(img_dir):
    lst = []
    length = len(os.listdir(img_dir))
    if os.path.exists(os.path.join(img_dir, '.ipynb_checkpoints')):
        length -= 1
    
    for i in range(1, length + 1):
        pic = str(i) + '.png'
        lst.append(os.path.join(img_dir, pic))
    
    return lst




def random_crop(feature, label, height, width):
    rect = transforms.RandomCrop.get_params(feature, (height, width))
    feature = functional.crop(feature, *rect)
    label = functional.crop(label, *rect)
    return feature, label


def elastic_transform(feature, label, sigma=10, points=3):
    x = np.array(feature)
    y = np.array(label)
    x, y = elasticdeform.deform_random_grid(X=[x.transpose(2, 0, 1), y],
                                            axis=[(1, 2), (0, 1)],
                                            sigma=sigma, points=points)
    x = x.transpose(1, 2, 0)
    y = np.where(y > 10, 255, 0)

    return x, y


class SegDataSet(Dataset):
    def __init__(self, lst, img_dir, mask_dir, crop_size, multi=False, mode="train"):

        self.img_dir = '/root/autodl-tmp/Blood/img'
        self.mask_dir = '/root/autodl-tmp/Blood/mask'

        if multi is True:
            self.img_list = [os.path.join(self.img_dir, img) for img in lst]
            self.mask_list = [os.path.join(self.mask_dir, mask) for mask in lst]
        else:
            self.img_list = load_images(img_dir)
            self.mask_list = load_images(mask_dir)
        
        self.crop_size = crop_size
        self.mode = mode
        
        
        print(f"read {len(self.img_list)} images")
        print(f"read {len(self.mask_list)} masks")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img = transforms.Resize(self.crop_size)(Image.open(self.img_list[item]))
        mask = transforms.Resize(self.crop_size)(Image.open(self.mask_list[item]))
        if self.mode == 'train':
            img, mask = elastic_transform(img, mask, sigma=7)
        else:
            img = np.array(img)
            mask = np.array(mask)
            mask = np.where(mask > 127, 255, 0)
        img = transforms.ToTensor()(img / 255)
        mask = transforms.ToTensor()(mask / 255).long()
        
        return img, mask
        

if __name__ == "__main__":

    a, b = get_kfold('/root/autodl-tmp/Blood/img', k=5)
    dt = SegDataSet(a[0], '/root/autodl-tmp/Blood/train/img', '/root/autodl-tmp/Blood/train/mask', (224, 224))
    loader = DataLoader(dt, batch_size=12, shuffle=True)
    for i, (x, y) in enumerate(loader):
        print(x.shape, y.shape)
        print(x)


