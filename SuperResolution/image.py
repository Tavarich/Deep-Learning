import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

image_path = '/root/autodl-tmp/DIV2K/DIV2K_train_HR/0002.png'

img1 = Image.open(image_path)  # PIL 对象，   RGB
img2 = cv2.imread(image_path)  # numpy 数组， BGR  (0, 255)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

img1_np = np.array(img1)  # numpy 数组 ，(H, W, C)  (0, 255)
img1_ts = to_tensor(img1_np)  # tensor 张量，(C, H, W)  (0, 1)

img1_pil = to_pil(img1_ts)  # PIL 对象，   RGB

print(img1_np.shape)
print(img1_ts.shape)
print(type(img1_pil))

if __name__ == "__main__":
    plt.subplot(1, 3, 1)
    plt.title('pil numpy array')
    plt.imshow(img1_np)

    plt.subplot(1, 3, 2)
    plt.title('cv numpy array')
    plt.imshow(img2)

    plt.subplot(1, 3, 3)
    plt.title('pil image')
    plt.imshow(img1_pil)

    plt.show()
