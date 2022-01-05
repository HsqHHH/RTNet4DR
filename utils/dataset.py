#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2



class multilesionDataset(Dataset):

    def __init__(self, image_paths, mask_paths=None, vessel_paths=None, transform=None):
        # image_paths : List of paths of images;
        # vessel_paths : List of paths of pesudo vessel masks;
        # mask_paths : List of lists containing four paths of lesion masks, i.e., EX, HE, MA, SE in order.
        assert len(image_paths) == len(mask_paths)
        self.t = transforms.ToTensor()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.vessel_paths = vessel_paths
        self.masks = []
        self.images = []
        self.vessels = []
        self.transform = transform
        self.get_size = []
        if self.mask_paths is not None:
            for image_path, mask_path, vessel_path in zip(image_paths, mask_paths, vessel_paths):
                self.images.append(self.pil_loader(image_path)[0])
                self.vessels.append(self.pil_loader(vessel_path)[0])
                self.get_size.append(self.pil_loader(image_path)[-1])
                mask = np.zeros(self.get_size[-1]).reshape(self.get_size[-1][-1],-1)
                for i,path in enumerate(mask_path):
                    lesion = self.cv2_loader(path)
                    mask += lesion * (i+1)
                self.masks.append(Image.fromarray(mask))

    def __len__(self):
        return len(self.image_paths)

    def pil_loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            h, w = img.size
            return [img.convert('RGB'), (h, w)]

    def cv2_loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = cv2.imread(f,0)
            return img

    def __getitem__(self, idx):

        info = [self.images[idx]]
        if self.mask_paths:
            info.append(self.masks[idx])
        if self.vessel_paths:
            info.append(self.vessels[idx])
        if self.transform:
            info = self.transform(info)
        size = self.get_size[idx]
        inputs = np.array(info[0])
        if inputs.shape[2] == 3:
            inputs = np.transpose(np.array(info[0]), (2, 0, 1))
            inputs = inputs / 255.
        mask = np.array(info[1])[:, :, 0]
        masks = np.array([mask])
        vessel = np.array(info[2])[:, :, 0]
        ret, vessels = cv2.threshold(vessel, 20, 255, 0)
        vessels = np.array([vessels], dtype=np.float)
        # vessels = np.array(vessels,dtype=np.float).reshape(-1,vessels.shape[0],vessels.shape[1])
        vessels = vessels / 255.
        return inputs, masks, vessels, size
