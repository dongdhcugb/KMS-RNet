import cv2

from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.dem_transform import trans_dem_gdal, trans_img_RS

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img_all = cv2.imread(os.path.join(self.root, id.split(' ')[0]), cv2.IMREAD_UNCHANGED)
        img = Image.fromarray(img_all[:, :, 0:3])
        dem = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[2]))))

        if self.mode == 'val':
            img, dem, mask = normalize(img, dem, mask)

            return img, dem, mask, id

        img, dem, mask = resize(img, dem, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, dem, mask = crop(img, dem, mask, self.size, ignore_value)
        img, dem, mask = hflip(img, dem, mask, p=0.5)

        if self.mode == 'train_l':

            if random.random() < 0.5:
                dem = trans_dem_gdal(np.array(dem))
            if random.random() < 0.5:
                img = Image.fromarray(trans_img_RS(img_all))

            return normalize(img, dem, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.5:
            dem_W = trans_dem_gdal(np.array(dem))
        else:
            dem_W = dem

        if random.random() < 0.5:
            img_w = Image.fromarray(trans_img_RS(img_all))

        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, dem_W, ignore_mask = normalize(img_s1, dem_W, ignore_mask)

        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        img_w, dem, _ = normalize(img_w, dem)

        return img_w, img_s1, img_s2, dem_W, dem, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
