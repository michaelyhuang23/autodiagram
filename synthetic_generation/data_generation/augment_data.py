import os
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.append('../../data_aug_experiment/')
from augmentations import *

image_dir = '../data/img_files/'
tex_dir = '../data/tex_files/'

num_imgs = max(int(file.split('.')[0]) for file in os.listdir(image_dir) if '.jpg' in file) + 1

next_id = num_imgs

def augment(img):
    img = torch.from_numpy(img[None, :, :])
    amp = np.random.uniform(0, 0.002)
    img = vert_curve_image_raw(img, amp, 255)
    img = hori_curve_image_raw(img, amp, 255)
    rotator = transforms.RandomRotation(10, fill=255)
    img = rotator(img)
    img = img.permute(1,2,0).numpy()
    return img


for i in range(num_imgs):
    o_img = cv2.imread(os.path.join(image_dir, f'{i}.jpg'))
    o_img = cv2.cvtColor(o_img, cv2.COLOR_BGR2GRAY)

    for j in range(2):
        img = augment(o_img)

        cv2.imwrite(os.path.join(image_dir, f'{next_id}.jpg'), img)
        with open(os.path.join(tex_dir, f'{i}.tex'), 'r') as f:
            file_data = f.read()
        with open(os.path.join(tex_dir, f'{next_id}.tex'), 'w') as f:
            f.write(file_data)
        next_id += 1



