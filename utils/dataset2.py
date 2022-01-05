from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize
import PIL
from PIL import Image
import os

from .transform import Compose, RandomCrop, RandomRotation, RandomVerticalFlip,RandomHorizontalFlip
from .preprocess import *
# from .extract_patches import create_patch_idx



class TrainDataset(Dataset):
    def __init__(self, imgs,masks,fovs,vessels,patches_idx,mode,args):
        self.imgs = imgs
        self.masks = masks
        self.fovs = fovs
        self.vessels = vessels
        self.patch_h, self.patch_w = args.train_patch_height, args.train_patch_width
        self.patches_idx = patches_idx
        self.inside_FOV = args.inside_FOV
        self.transforms = None
        if mode == "train":
            self.transforms = Compose([
                # RandomResize([56,72],[56,72]),
                RandomCrop((48,48)),
                # RandomFlip_LR(prob=0.5),
                # RandomFlip_UD(prob=0.5),
                # RandomRotate()
                RandomRotation(10),
                RandomVerticalFlip(),
                RandomHorizontalFlip()
            ])

    def __len__(self):
        return len(self.patches_idx)

    def __getitem__(self, idx):
        n, x_center, y_center = self.patches_idx[idx]

        data = self.imgs[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]
        mask = self.masks[n,:,y_center-int(self.patch_h/2):y_center+int(self.patch_h/2),x_center-int(self.patch_w/2):x_center+int(self.patch_w/2)]
        vessel = self.vessels[n, :, y_center - int(self.patch_h / 2):y_center + int(self.patch_h / 2),
               x_center - int(self.patch_w / 2):x_center + int(self.patch_w / 2)]

        if self.transforms:
            data = Image.fromarray((np.squeeze(data)*255.).astype(np.uint8))
            mask = Image.fromarray((np.squeeze(mask)*255.).astype(np.uint8))
            vessel = Image.fromarray((np.squeeze(vessel)*255.).astype(np.uint8))

            data, mask, vessel = self.transforms([data, mask, vessel])
            data = np.expand_dims(np.array(data),0)
            mask = np.expand_dims(np.array(mask),0)
            vessel = np.expand_dims(np.array(vessel),0)
            data = data / 255.
            mask = mask / 255.
            vessel = vessel / 255.

        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).long()
        vessel = torch.from_numpy(vessel).long()


        # if self.transforms:
        #     data, mask = self.transforms(data, mask)

        return data, mask.squeeze(0), vessel.squeeze(0)

def data_load(file_path):
    img_list = []
    gt_list = []
    fov_list = []
    vessel_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # read a line
            if not lines:
                break
            img, gt, fov, vessel = lines.split(' ')
            img_list.append(img)
            gt_list.append(gt)
            fov_list.append(fov)
            vessel_list.append(vessel)

    imgs = None
    groundTruth = None
    FOVs = None
    VS = None
    for i in range(len(img_list)):
        img = np.asarray(PIL.Image.open(img_list[i])) #0-255
        gt = np.asarray( PIL.Image.open(gt_list[i])) # 0,1,2,3,4,5
        vs = np.asarray( PIL.Image.open(vessel_list[i])) # 0,255
        # import pdb
        # pdb.set_trace()
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        if len(vs.shape) == 3:
            vs = vs[:, :, 0]
        fov = np.asarray(PIL.Image.open(fov_list[i])) # 0,255
        if len(fov.shape) == 3:
            fov = fov[:, :, 0]
        imgs = np.expand_dims(img, 0) if imgs is None else np.concatenate((imgs, np.expand_dims(img, 0)))
        groundTruth = np.expand_dims(gt, 0) if groundTruth is None else np.concatenate(
            (groundTruth, np.expand_dims(gt, 0)))
        VS = np.expand_dims(vs, 0) if VS is None else np.concatenate(
            (VS, np.expand_dims(vs, 0)))
        FOVs = np.expand_dims(fov, 0) if FOVs is None else np.concatenate((FOVs, np.expand_dims(fov, 0)))

    # assert (np.min(FOVs) == 0 and np.max(FOVs) == 255)
    # assert ((np.min(groundTruth) == 0 and (
    #             np.max(groundTruth) == 255 or np.max(groundTruth) == 1)))  # CHASE_DB1数据集GT图像为单通道二值（0和1）图像
    # if np.max(groundTruth) == 1:
    #     print("\033[0;31m Single channel binary image is multiplied by 255 \033[0m")
    #     groundTruth = groundTruth * 255 #
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    groundTruth = np.expand_dims(groundTruth, 1)
    FOVs = np.expand_dims(FOVs, 1)
    VS = np.expand_dims(VS, 1)

    print('ori data shape < ori_imgs:{} GTs:{} FOVs:{}'.format(imgs.shape, groundTruth.shape, FOVs.shape))
    print("imgs pixel range %s-%s: " % (str(np.min(imgs)), str(np.max(imgs))))
    print("GTs pixel range %s-%s: " % (str(np.min(groundTruth)), str(np.max(groundTruth))))
    print("FOVs pixel range %s-%s: " % (str(np.min(FOVs)), str(np.max(FOVs))))
    print("==================data have loaded======================")
    return imgs, groundTruth, FOVs, VS

def data_preprocess(data_path_list):

    train_imgs_original, train_masks, train_FOVs, train_VS = data_load(data_path_list)
    assert (len(train_imgs_original.shape) == 4)
    assert (train_imgs_original.shape[1] == 3)  # Use the original images
    # black-white conversion
    train_imgs = rgb2gray(train_imgs_original)
    # my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.  # reduce to 0-1 range
    # train_masks = train_masks // 255
    train_FOVs = train_FOVs // 255
    train_VS = train_VS // 255
    return train_imgs, train_masks, train_FOVs, train_VS




def get_TrainDataset(args):
    imgs_train, masks_train, fovs_train, vessel_train = data_preprocess(data_path_list = args.train_data_path_list)

    patches_idx = create_patch_idx(fovs_train, args)
    train_idx,val_idx = np.vsplit(patches_idx, (int(np.floor((1-args.val_ratio)*patches_idx.shape[0])),))
    train_set = TrainDataset(imgs_train, masks_train, fovs_train, vessel_train, train_idx,mode="train",args=args)
    val_set = TrainDataset(imgs_train, masks_train, fovs_train, vessel_train, val_idx,mode="val",args=args)

    return train_set,val_set

def is_patch_inside_FOV(x,y,fov_img,patch_h,patch_w,mode='center'):
    """
    check if the patch is contained in the FOV,
    The center mode checks whether the center pixel of the patch is within fov,
    the all mode checks whether all pixels of the patch are within fov.
    """
    if mode == 'center':
        return fov_img[y,x]
    elif mode == 'all':
        fov_patch = fov_img[y-int(patch_h/2):y+int(patch_h/2),x-int(patch_w/2):x+int(patch_w/2)]
        return fov_patch.all()
    else:
        raise ValueError("\033[0;31mmode is incurrent!\033[0m")

def create_patch_idx(img_fovs, args):
    assert len(img_fovs.shape)==4
    N,C,img_h,img_w = img_fovs.shape
    res = np.empty((args.N_patches,3),dtype=int)
    print("")

    seed=2021
    count = 0
    while count < args.N_patches:
        random.seed(seed) # fuxian
        seed+=1
        n = random.randint(0,N-1)
        x_center = random.randint(0+int(args.train_patch_width/2),img_w-int(args.train_patch_width/2))
        y_center = random.randint(0+int(args.train_patch_height/2),img_h-int(args.train_patch_height/2))

        #check whether the patch is contained in the FOV
        if args.inside_FOV=='center' or args.inside_FOV == 'all':
            if not is_patch_inside_FOV(x_center,y_center,img_fovs[n,0],args.train_patch_height,args.train_patch_width,mode=args.inside_FOV):
                continue
        res[count] = np.asarray([n,x_center,y_center])
        count+=1

    return res