import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from models import Generator
from PIL import Image


def predict(model, img, scaling_size=2, mode='gpu'):
    """
    :param model: 神经网络模型
    :param img:   待超分图片，PIL Image
    :param scaling_size: 放大倍率
    :param mode:  使用GPU或CPU推理
    :return: None
    """
    
    device = torch.device('cuda' if mode == 'gpu' else 'cpu')
    model.to(device)
    model.eval()
    
    x = transforms.ToTensor()(img)
    x.unsqueeze_(0)
    x = x.to(device)
    
    sr_tensor = model(x).squeeze(0).cpu().detach()
    sr_tensor = torch.clamp(sr_tensor, 0, 1)  # 裁剪超出0——1范围的像素值
    sr_img = transforms.ToPILImage()(sr_tensor)

    sr_img.save('/root/CV/SuperResolution/srgan/result/srgan.jpg')

    bi_img = img.resize((int(img.width * scaling_size), int(img.height * scaling_size)), Image.BICUBIC)
    bi_img.save('/root/CV/SuperResolution/srgan/result/bicubic.jpg')


def main():
    img_path = "/root/autodl-tmp/DIV2K/test/girl_lr.jpg"
    img = Image.open(img_path)
    
    save_path = "/root/autodl-tmp/Models/SuperResolution/srgan"
    checkpoint = torch.load(save_path, map_location='cpu')

    model = Generator(scaling_size=2)
    model.load_state_dict(checkpoint['modelG'])

    predict(model, img)


if __name__ == "__main__":
    main()
