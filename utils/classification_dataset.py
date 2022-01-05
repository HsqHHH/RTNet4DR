from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import random
import torch.nn.functional as F
import glob
import os
import cv2


def getPathList(root):
    files = os.listdir(root)
    res = []
    for file in files:
        # print(os.path.join(root, file, '*'))
        img_paths = glob.glob(os.path.join(root,file,'*'))
        res.extend(img_paths)
    random.shuffle(res)
    return res



class classDataset(Dataset):
    def __init__(self,data_list):
        super(classDataset, self).__init__()
        self.data_list = data_list
        self.images, self.labels = self.load_data()

    def __len__(self):
        return len(self.images)


    def load_data(self):
        images = []
        labels = []
        for path in self.data_list:
            img = cv2.imread(path,0)
            print(img.shape)
            if path.split('\\')[-2] == 'normal':
                label = 0
            elif path.split('\\')[-2] == 'abnormal':
                label = 1
            else:
                raise TypeError('Input has no valid label.')
            images.append(img)
            labels.append(label)
        return images,labels

    def __getitem__(self, ind):
        img = self.images[ind]
        lab = self.labels[ind]
        return img,lab





if __name__ == '__main__':
    root = ''
    data_list = getPathList(root)
    dataset = classDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=12)

