import cv2
import os
from PIL import Image
import numpy as np
import csv
from random import random, uniform, randint, randrange, choice, sample, shuffle
import glob
import torch

train_ratio = 0.8

def get_images(image_dir, preprocess='0', phase='train'):
    #image_dir = '/home1/user/huangshiqi/data/newlesion'
    if phase == 'train':
        setname = 'train'
    elif phase == 'valid':
        setname = 'train'
    elif phase == 'test':
        setname = 'test'

    img_dir = os.path.join(image_dir, setname, 'image')
    label_dir = os.path.join(image_dir, setname, 'label')
    vessel_dir = os.path.join(image_dir, setname, 'vessel')

    ps = []
    ns = []
    num_MA, num_HE, num_EX, num_SE = 0,0,0,0
    import csv
    with open(os.path.join(image_dir, setname, setname + '.csv'), 'r') as csvfile: 
        reader = csv.reader(csvfile)
        for info in reader:
            if info[1] == 'positive':
                continue
            if info[1] == 'True':
                ps.append(info[0])
                if info[2] != '0': num_MA += 1
                if info[3] != '0': num_HE += 1
                if info[4] != '0': num_EX += 1
                if info[5] != '0': num_SE += 1
            else:
                ns.append(info[0])


    ps.sort()
    ns.sort()
    # import pdb
    # pdb.set_trace()
    train_number = int(len(ps) * train_ratio)
    valid_number = int(len(ps) * (1-train_ratio))
    if phase == 'train':
        image_positive = ps[:train_number]
    elif phase == 'valid':
        image_positive = ps[train_number:]
    else:
        image_positive = ps

    negtive_number = int(len(image_positive) * 10)
    shuffle(ns)
    image_negtive = ns[:negtive_number]
    images = image_positive + image_negtive
    shuffle(images)

    image_paths = []
    label_paths = []
    vessel_paths = []
    for img in images:
        image_path = os.path.join(img_dir,img+'.jpg')
        label_path = os.path.join(label_dir,img+'.tif')
        vessel_path = os.path.join(vessel_dir,img+'.tif')
        assert os.path.exists(image_path), 'image error'
        assert os.path.exists(label_path), 'label error'
        assert os.path.exists(vessel_path), 'vessel error'


        image_paths.append(image_path)
        label_paths.append(label_path)
        vessel_paths.append(vessel_path)

    return image_paths, label_paths, vessel_paths


def get_images_IDRiD(image_dir, preprocess='0', phase='train'):
    # image_dir = '/home1/user/huangshiqi/data/lesion'
    lesions = {'EX':'Hard Exduate','HE':'Haemorrohage','MA':'Microaneurysm','SE':'Soft Exduate'}

    if phase == 'train':
        setname = 'train'
    elif phase == 'valid':
        setname = 'train'
    elif phase == 'test':
        setname = 'test'


    limit = 2
    grid_size = 8
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess))
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname))

        # compute mean brightess
        meanbright = 0.
        images_number = 0
        for tempsetname in ['train', 'test']:
            imgs_ori = glob.glob(os.path.join(image_dir, 'Images/' + tempsetname + '/*.jpg'))
            imgs_ori.sort()
            images_number += len(imgs_ori)
            # mean brightness.
            for img_path in imgs_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, 'Groundtruths', tempsetname, 'Mask', img_name + '_MASK.tif')
                gray = cv2.imread(img_path, 0)
                mask_img = cv2.imread(mask_path, 0)
                brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                meanbright += brightness
        meanbright /= images_number

        imgs_paths = glob.glob(os.path.join(image_dir, 'Images/' + setname + '/*.jpg'))

        preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None],
                           '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright],
                           '6': [True, True, None], '7': [True, True, meanbright]}
        for img_path in imgs_paths:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            mask_path = os.path.join(image_dir, 'Groundtruths', setname, 'Mask', img_name + '_MASK.tif')
            clahe_img = clahe_gridsize(img_path, mask_path, denoise=preprocess_dict[preprocess][0],
                                       contrastenhancement=preprocess_dict[preprocess][1],
                                       brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit,
                                       gridsize=grid_size)
            cv2.imwrite(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname, os.path.split(img_path)[-1]),
                        clahe_img)

    imgs = glob.glob(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname, '*.jpg'))
    imgs.sort()
    vessel_path = os.path.join(image_dir, 'Groundtruths', setname, 'vessel')
    vessel_paths = glob.glob(os.path.join(vessel_path,'*.tif'))
    vessel_paths.sort()
    train_number = int(len(imgs) * train_ratio)
    eval_number = int(len(imgs) * (1 - train_ratio))
    if phase == 'train':
        image_paths = imgs[:train_number]
    elif phase == 'eval':
        image_paths = imgs[train_number:]
    else:
        image_paths = imgs
    mask_paths = []
    for image_path in image_paths:
        lesion_path4 = []
        name = os.path.split(image_path)[1].split('.')[0]

        mask_paths.append(candidate1_path)
        candidate_vessel_path = os.path.join(vessel_path, name + '_vessel' + '.tif')
        vessel_paths.append(candidate_vessel_path)
        for le in lesions:
            candidate_lesion_path = os.path.join(mask_path, lesions[le], name + '_' + le + '.tif')
            if os.path.exists(candidate_lesion_path):
                lesion_path4.append(candidate_lesion_path)
            else:
                im = cv2.imread(candidate_vessel_path)
                cv2.imwrite(candidate_lesion_path, torch.zeros_like(im))
                lesion_path4.append(candidate_lesion_path)
        mask_paths.append(lesion_path4)

    return image_paths, mask_paths, vessel_paths


def get_images_DDR(image_dir, preprocess='0', phase='train'):
    lesions = {'EX': 'Hard Exduate', 'HE': 'Haemorrohage', 'MA': 'Microaneurysm', 'SE': 'Soft Exduate'}
    if phase == 'train':
        setname = 'train'
    elif phase == 'valid':
        setname = 'valid'
    elif phase == 'test':
        setname = 'test'
    image_dir = os.path.join(image_dir,setname)

    limit = 2
    grid_size = 8
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess))
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess))

        # compute mean brightess
        meanbright = 0.
        images_number = 0
        imgs_ori = glob.glob(os.path.join(image_dir, 'image' + '/*.jpg'))
        imgs_ori.sort()
        images_number += len(imgs_ori)
        # mean brightness.
        for img_path in imgs_ori:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            mask_path = os.path.join(image_dir, 'label', image_dir, 'Mask', img_name + '_MASK.tif')
            gray = cv2.imread(img_path, 0)
            mask_img = cv2.imread(mask_path, 0)
            brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
            meanbright += brightness
        meanbright /= images_number

        imgs_paths = glob.glob(os.path.join(image_dir, 'image' + '/*.jpg'))

        preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None],
                           '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright],
                           '6': [True, True, None], '7': [True, True, meanbright]}
        for img_path in imgs_paths:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            mask_path = os.path.join(image_dir, 'label', 'Mask', img_name + '_MASK.tif')
            clahe_img = clahe_gridsize(img_path, mask_path, denoise=preprocess_dict[preprocess][0],
                                       contrastenhancement=preprocess_dict[preprocess][1],
                                       brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit,
                                       gridsize=grid_size)
            cv2.imwrite(os.path.join(image_dir, 'Images_CLAHE' + preprocess, os.path.split(img_path)[-1]),
                        clahe_img)

    imgs = glob.glob(os.path.join(image_dir, 'Images_CLAHE' + preprocess,  '*.jpg'))
    imgs.sort()
    vessel_path = os.path.join(image_dir, 'label',  'vessel')
    vessel_paths = glob.glob(os.path.join(vessel_path, '*.tif'))
    vessel_paths.sort()
    train_number = int(len(imgs) * train_ratio)
    eval_number = int(len(imgs) * (1 - train_ratio))
    image_paths = imgs
    mask_paths = []
    for image_path in image_paths:
        lesion_path4 = []
        name = os.path.split(image_path)[1].split('.')[0]
        candidate_vessel_path = os.path.join(vessel_path, name + '_vessel' + '.tif')
        vessel_paths.append(candidate_vessel_path)
        for le in lesions:
            candidate_lesion_path = os.path.join(mask_path, le, name + '.tif')
            if os.path.exists(candidate_lesion_path):
                lesion_path4.append(candidate_lesion_path)
            else:
                im = cv2.imread(candidate_vessel_path)
                cv2.imwrite(candidate_lesion_path, torch.zeros_like(im))
                lesion_path4.append(candidate_lesion_path)
        mask_paths.append(lesion_path4)

    return image_paths, mask_paths, vessel_paths

class AverageMeter(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count