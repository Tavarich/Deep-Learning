import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

program_path = "classify-birds/"
train_path = program_path + "train_set"
test_path = program_path + "test_set"
classes_path = program_path + "classes.txt"


def get_image_list(path):
    """
    :param path:存放图片的路径
    :return: list, 存放图片的名称
    """
    lst = []
    for img in os.listdir(path):
        lst.append(img)  # 001.Black_footed_Albatross_112.jpg if train  else 1.jpg
    return lst


def get_image_list_csv(path):
    """
    :param path: csv文件的路径
    :return: list, 存放图片的名称
    """
    lst = []
    with open(path + "test.csv", "r") as f:
        reader = csv.reader(f)
        for (i, row) in enumerate(reader):
            lst.append(row[0])  # 地址的一部分
    return lst


def get_match_dict(class_path):
    """
    :param class_path:类别文件的路径
    :return: 类别和对应标号的字典
    """
    map = {}
    with open(class_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label = int(line.split(" ")[0])
            name = line.split(" ")[1].split(".")[1]
            map[label] = name
    return map


def get_kfold(image_list, k):
    """
    :param image_list: 图片列表
    :param k:交叉验证的折数
    :return:验证集，数据集的图片列表
    """
    indexes = [int(img.split('.')[0]) - 1 for img in image_list]
    skf = StratifiedKFold(n_splits=k)
    train_kfold = []
    valid_kfold = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_list, indexes)):
        train_kfold.append([image_list[i] for i in train_idx])
        valid_kfold.append([image_list[j] for j in val_idx])
    return train_kfold, valid_kfold


class BirdDataSet(Dataset):
    def __init__(self, image_list, image_folder, mode='train'):
        """
        :param image_list:    图片列表
        :param image_folder:  图片所在文件夹
        :param mode:          数据集模式，训练或者验证
        """
        super().__init__()
        self.image_list = image_list
        self.image_folder = image_folder
        self.mode = mode
        self.train_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.valid_trans = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        if self.mode == 'train':
            image = self.train_trans(image)
        else:
            image = self.valid_trans(image)
        label = int(self.image_list[idx].split('.')[0]) - 1
        return image, label


match_dict = get_match_dict(classes_path)

if __name__ == "__main__":

    train_list = get_image_list(train_path)
    test_list = get_image_list(test_path)
    print(train_list[:5])
    print(test_list[:5])
    
