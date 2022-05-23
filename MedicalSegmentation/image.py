import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == '__main__':
    test = '../test/test'
    if not os.path.exists(test):
        os.makedirs(test)

    img_path = '../train/mask/1.png'
    img = Image.open(img_path)
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    img = np.asarray(img)
    print(img.shape)
    _, mask = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
    print(mask.shape)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
    cv2.imwrite(test + '/tst.jpg', mask)