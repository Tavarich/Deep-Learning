import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SrDataSet(Dataset):
    def __init__(self, image_path, crop_size, scaling_size):
        """
        :param image_path: 数据集图片路径
        :param crop_size: 图片裁剪的尺寸
        :param scaling_size: 重构倍率
        :return: None
        """
        self.image_list = []
        self.image_path = image_path
        self.crop_size = crop_size
        self.scaling_size = scaling_size

        for img in os.listdir(image_path):
            if img.split('.')[-1] != 'png':
                continue
            self.image_list.append(img)
        self.pre_treatment = transforms.Compose([
            transforms.RandomCrop(crop_size),  # 在HR图片中裁剪一个子块进行训练
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        self.x_treatment = transforms.Compose([
            transforms.Resize(crop_size // scaling_size),  # 通过Resize进行下采样（默认双线性插值）
            transforms.ToTensor(),  # 不使用归一化
        ])
        self.y_treatment = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        img = Image.open(image_path).convert('RGB')
        img = self.pre_treatment(img)
        lr = self.x_treatment(img)
        hr = self.y_treatment(img)

        return lr, hr


train_path = '/root/autodl-tmp/DIV2K/DIV2K_train_HR_560'

if __name__ == "__main__":
    train_set = SrDataSet(train_path, 448, 2)
    lr, hr = train_set[0]
    print(lr.shape, hr.shape)
