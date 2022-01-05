#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import torch
import torch.nn.functional as F
from utils.utils import get_images_IDRiD
from utils.dataset import multilesionDataset
from torch.utils.data import DataLoader
import argparse
# from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def eval_model(model_choice, model, eval_loader, args):
    model.eval()
    les = ['EX', 'HE', 'MA', 'SE']  # denote the lesion GT, not the model_choice
    masks_soft = []
    masks_hard = []
    store = []
    with torch.no_grad():
        num = 0
        for inputs, true_masks, vessel in eval_loader:

            num += 1
            print('test the %d image' % num)
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = torch.where(true_masks > 0, torch.ones_like(true_masks), torch.zeros_like(true_masks))
            true_masks = true_masks.to(device=device, dtype=torch.float)
            vessel = vessel.to(device=device, dtype=torch.float)
            bs, _, h, w = inputs.shape
            # not ignore the last few patches
            masks_pred = model(inputs)

            masks_pred_softmax_batch = F.softmax(masks_pred, dim=1).cpu().numpy()
            masks_soft_batch = masks_pred_softmax_batch[:, 1:, :, :]
            masks_hard_batch = true_masks.cpu().numpy()

            masks_soft.extend(masks_soft_batch)
            masks_hard.extend(masks_hard_batch)
           for ind in range(len(les)):
               masks_soft[les[ind]].extend(masks_soft_batch[:,ind:ind+1,:,:])
               masks_hard[les[ind]].extend(masks_hard_batch==(ind+1))
               if not os.path.exists(os.path.join(save_dir,les[ind])):
                   os.mkdir(os.path.join(save_dir,les[ind]))
               tmp_mask = masks_soft_batch[0,ind,:,:]
               tmp_mask_hard = torch.where(torch.Tensor(tmp_mask)>0.55, torch.ones_like(torch.Tensor(tmp_mask)), torch.zeros_like(torch.Tensor(tmp_mask)))
               scipy.misc.imsave(os.path.join(save_dir, les[ind], 'soft_'+str(num).zfill(2)+'.jpg'), tmp_mask)
               scipy.misc.imsave(os.path.join(save_dir, les[ind], 'hard_'+str(num).zfill(2)+'.jpg'), tmp_mask_hard.numpy())
            del masks_pred_softmax_batch, masks_soft_batch, masks_hard_batch
    masks_soft = np.array(masks_soft).transpose((1, 0, 2, 3))
    masks_hard = np.array(masks_hard).transpose((1, 0, 2, 3))
    masks_soft = np.reshape(masks_soft, (masks_soft.shape[0], -1))
    masks_hard = np.reshape(masks_hard, (masks_hard.shape[0], -1))
    masks_true = masks_hard[0]
    masks_score = masks_soft[0]
    del masks_soft, masks_hard

    x = torch.sort(torch.randint(0, 329730048, (10000,)))[0]
    masks_true = masks_true[x]
    masks_score = masks_score[x]

    precision, recall, _ = precision_recall_curve(masks_true, masks_score)
    fpr, tpr, _ = roc_curve(masks_true, masks_score)
    auc_pr = auc(recall, precision)
    auc_roc = auc(fpr, tpr)

    del precision, recall, fpr, tpr
    print('%6f\n%6f\n' % (auc_pr, auc_roc))
    return store
    # writer.add_scalar(args.lesion+'_PR',precision,recall)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--preprocess', type=str, default='7')
    parser.add_argument('--model', type=str)
    parser.add_argument('--lesion', type=str, default='EX')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--img_dir', type=str, default='/data3/huangsq/lesion/')
    parser.add_argument('--img_dir_test', type=str, default='/data3/huangsq/lesion/')
    # model params
    parser.add_argument('--backbone', type=str, default='densenet161')
    parser.add_argument('--encoder_weights', type=str, default='imagenet')
    parser.add_argument('--lesion_classes', type=int, default=2)
    parser.add_argument('--vessel_classes', type=int, default=2)
    parser.add_argument('--activate', default=None, choices=[None, 'relu', 'softmax', 'sigmoid'])
    args = parser.parse_args()
    # Set random seed for Pytorch and Numpy for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    # import model
    import DR_segmentation as DRseg
    model = DRseg.Unet(encoder_name=args.backbone, encoder_weights=args.encoder_weights,
                     lesion_classes=args.lesion_classes)
    model_choice = ['all', 'EX', 'HE', 'MA', 'SE']

    # prepare dataloader
    test_image_paths, test_mask_paths, test_vessel_paths = get_images_IDRiD(args.img_dir_test, args.preprocess, phase='test')
    test_dataset = multilesionDataset(test_image_paths, test_mask_paths, test_vessel_paths)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)

    # saved information
    save_dir = ""
    model_name = "model_"

    # metric storage
    metric = np.random.randn(5, 8)  # (all,EX,HE,MA,SE)*(pr,roc/EX,HE,MA,SE)

    # test 'all','EX','HE','MA','SE' models respectively!
    for i in range(len(model_choice)):
        if i == 0:
            print('Start to test the ' + model_choice[i] + ' model!')
            # resume = args.model
            resume = os.path.join(save_dir, model_name + model_choice[i] + ".pth.tar")
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                model.load_state_dict(checkpoint['state_dict'])
                print('Model loaded from {}'.format(resume))
            else:
                print("=> no checkpoint found at '{}'".format(resume))
            model.cuda()
            store = eval_model(model_choice[i], model, test_loader, args)
            metric[i, :] = np.array(store)
            torch.cuda.empty_cache()

   csvf_metric = os.path.join(save_dir,'metric.csv')
   with open(csvf_metric,'w') as csvfile:
       writer = csv.writer(csvfile)
       for row in range(metric.shape[0]):
           writer.writerow(metric[row,:])
       writer.writerow(np.argmax(metric,axis=0))
#
#    # to save the best lesion GT
#    index = np.argmax(metric[:,[0,2,4,6]],axis=0)
#    model_saveimg = [] #best auc_pr model for EX,HE,MA,SE respectively
#    for tmp in index:
#        model_saveimg.append(model_choice[tmp])
#    for k in range(len(model_saveimg)):
#        print('Start to test the '+model_saveimg[k]+' model to save '+model_choice[k+1]+' img!')
#
#        # resume = args.model
#        resume = os.path.join(save_dir, model_name+model_saveimg[k]+".pth.tar")
#        if os.path.isfile(resume):
#            print("=> loading checkpoint '{}'".format(resume))
#            checkpoint = torch.load(resume,map_location='cuda:0')
#            model.load_state_dict(checkpoint['state_dict'])
#            print('Model loaded from {}'.format(resume))
#        else:
#            print("=> no checkpoint found at '{}'".format(resume))
#
#        model.cuda()
#        eval_model_saveimg(k, model, test_loader,args)





