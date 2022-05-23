import os
import random
import shutil
from PIL import Image
import cv2

def split_pics(img_path, list, dist):
    if not os.path.exists(dist):
        os.makedirs(dist)
    for (i, img) in enumerate(list):
        src = os.path.join(img_path, img)
        dst = os.path.join(dist, str(i + 1) + ".png")
        shutil.copy(src, dst)

        
if __name__ == "__main__":

    factor = 0.9
    lst = [str(i) + ".png" for i in range(1, 216)]
    random.shuffle(lst)
    train = lst[: int(len(lst) * factor)]
    valid = lst[int(len(lst) * factor) + 1:]
    img_path = "/root/autodl-tmp/Blood/img"
    mask_path = "/root/autodl-tmp/Blood/mask"

    split_pics(img_path, train, '/root/autodl-tmp/Blood/train/img')
    split_pics(mask_path, train, '/root/autodl-tmp/Blood/train/mask')

    split_pics(img_path, valid, '/root/autodl-tmp/Blood/valid/img')
    split_pics(mask_path, valid, '/root/autodl-tmp/Blood/valid/mask')


    