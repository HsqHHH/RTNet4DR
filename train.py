#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import random
# import copy
from sklearn.metrics import average_precision_score, roc_auc_score
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
# from unet import UNet
from config import parse_args
# from utils import get_images_IDRiD,
# from dataset import multilesionDataset
from utils.transform import *
from torch.utils.data import DataLoader
from utils.dataset2 import  get_TrainDataset
from utils.logger import Logger, Print_Logger
from utils.utils import AverageMeter
from collections import OrderedDict

# from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def eval_model(model, eval_loader):
    model.eval()
    masks_pred = []
    masks_gt = []
    ap = []
    auc = []

    with torch.set_grad_enabled(False):  # torch.no_grad()
        for inputs, true_masks, vessel in eval_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)
            true_masks = true_masks.unsqueeze(1)
            # import pdb
            # pdb.set_trace()
            bs, _, h, w = inputs.shape
            masks = model(inputs)
            # h_size = (h - 1) // args.img_size + 1
            # w_size = (w - 1) // args.img_size + 1
            # masks = torch.zeros(bs,args.lesion_classes,h,w)

            # for i in range(h_size):
            #     for j in range(w_size):
            #         h_max = min(h, (i + 1) * args.img_size)
            #         w_max = min(w, (j + 1) * args.img_size)
            #         h_min = min(h-args.img_size, i*args.img_size)
            #         w_min = min(w-args.img_size,j*args.img_size)
            #         inputs_part = inputs[:,:, h_min:h_max, w_min:w_max]
            #         masks_pred_single,_ = model(inputs_part) # 512*512 preds
            #         masks[:,:, h_min:h_max, w_min:w_max] = masks_pred_single # paste cropped part to whole image
            masks = F.softmax(masks, dim=1).cpu().numpy()
            masks_pred_batch = masks[:, 1:, :, :] #(4,1,2848,3488)
            masks_gt_batch = true_masks.cpu().numpy()
            masks_pred.extend(masks_pred_batch)
            masks_gt.extend(masks_gt_batch)

    masks_pred = np.array(masks_pred).transpose((1, 0, 2, 3))
    # print(masks_gt.shape)
    masks_gt = np.array(masks_gt).transpose((1, 0, 2, 3))
    masks_pred = np.reshape(masks_pred, (masks_pred.shape[0], -1))
    masks_gt = np.reshape(masks_gt, (masks_gt.shape[0], -1))

    # x = torch.sort(torch.randint(0, 329730048, (10000,)))[0]
    # masks_gt = masks_gt[x]
    # masks_pred = masks_pred[x]

    for i in range(4):
        ap_lesion = average_precision_score(masks_gt[0] == i+1, masks_pred[i])
        # print((masks_gt[0] == i+1).shape,masks_pred[i].shape)
        print((masks_gt[0] == i+1).max(),masks_pred[i].max())

        auc_lesion = roc_auc_score(masks_gt[0] == i+1, masks_pred[i])
        ap.append(ap_lesion)
        auc.append(auc_lesion)

    val_log = OrderedDict([('EX_ap',ap[0]),('HE_ap',ap[1]),('MA_ap',ap[2]),('SE_ap',ap[3]),
                           ('EX_auc',ap[0]),('HE_auc',ap[1]),('MA_auc',ap[2]),('SE_auc',ap[3])])

    return val_log




def step(optimizer,closure=None):
      """Performs a single optimization step.

      Arguments:
          closure (callable, optional): A closure that reevaluates the model
              and returns the loss.
      """
      loss = None
      if closure is not None:
          loss = closure()

      for group in optimizer.param_groups:
          weight_decay = group['weight_decay']
          momentum = group['momentum']
          dampening = group['dampening']
          nesterov = group['nesterov']

          for p in group['params']:
              if p.grad is None:
                  continue
              d_p = p.grad.data
              
              # if d_p.device.type == 'cpu':
                  # d_p = d_p.to(torch.device('cuda:'+args.gpu))
                  # print('turn cuda:d_p.device:',d_p.device.type)
                  # is_cpu(optimizer)
              if weight_decay != 0:
                  d_p.add_(weight_decay, p.data)
                  p_cpu(d_p)
              if momentum != 0:
                  param_state = optimizer.state[p]
                  if 'momentum_buffer' not in param_state:
                      buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                  else:
                      buf = param_state['momentum_buffer']
                      buf.mul_(momentum).add_(1 - dampening, d_p)
                  if nesterov:
                      d_p = d_p.add(momentum, buf)
                  else:
                      d_p = buf

              p.data.add_(-group['lr'], d_p)

      return loss


def train_model(model, train_loader, eval_loader, criterion2,criterion5, g_optimizer, g_scheduler, num_epochs=5, start_epoch=0, start_step=0):
    model.to(device=device)
    tot_step_count = start_step

    best_ap = 0.
    best_EX_ap,best_HE_ap,best_SE_ap,best_MA_ap = 0.,0.,0.,0.
    # save
    dir_checkpoint = os.path.join(args.outf, args.save)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print('Starting epoch {}/{}.\t\n'.format(epoch + 1, start_epoch+num_epochs))
        model.train()
        steps = 0
        start_time = time.time()
        train_loss = AverageMeter()
        train_single_lesion_loss = [AverageMeter() for i in range(4)]

        for images, lesions, vessels in train_loader:
            images = images.to(device=device, dtype=torch.float)
            bs = images.shape[0]
            lesions = lesions.to(device=device, dtype=torch.float)
            vessels = vessels.to(device=device, dtype=torch.float)
            # lesion_masks, vessel_masks = model(images)
            lesion_masks = model(images)

            lesion_masks_transpose = lesion_masks.permute(0, 2, 3, 1)
            lesion_masks_flat = lesion_masks_transpose.reshape(-1, lesion_masks_transpose.shape[-1])
            # vessel_masks_transpose = vessel_masks.permute(0, 2, 3, 1)
            # vessel_masks_flat = vessel_masks_transpose.reshape(-1, vessel_masks_transpose.shape[-1])

            true_masks_flat = lesions.reshape(-1) # [1048576]
            # vessel_flat = vessels.reshape(-1)

            loss_lesion = criterion5(lesion_masks_flat, true_masks_flat.long())
            # loss_vessel = criterion2(vessel_masks_flat, vessel_flat.long())
            loss_vessel = 0
            g_loss = loss_lesion + loss_vessel*0.1
            train_loss.update(g_loss.item(), bs)
            loss_single_lesion = torch.zeros(4,)
            for i in range(4):
                loss = criterion2(lesion_masks_flat[:,[0,(i+1)]], (true_masks_flat==(i+1)).long())
                train_single_lesion_loss[i].update(loss.item(), bs)
            steps += bs
            
            g_loss = loss_lesion
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            tot_step_count += 1
        end_time = time.time()
        print(end_time-start_time,'s')
        g_scheduler.step()
        train_log = OrderedDict([('train_loss', train_loss.avg),('EX_loss',train_single_lesion_loss[0].avg),('HE_loss',train_single_lesion_loss[1].avg),('MA_loss',train_single_lesion_loss[2].avg),('SE_loss',train_single_lesion_loss[3].avg),])

        if (epoch) % 5 == 0:
            val_log = eval_model(model, val_loader)
            # print('[epoch: %3d]lesion_loss: %6f vessel_loss:%6f' % (epoch+1,loss_lesion.item(),loss_vessel.item()))
            lesions = ['EX','HE','MA','MA']
            print([i for i in zip(lesions,ap)])
            print([i for i in zip(lesions, auc)])
            if ap[0] > best_ap:
                print('save the model!')
                best_ap = ap[0]
                state = {
                    'epoch': epoch,
                    'step': tot_step_count,
                    'state_dict': model.state_dict(),
                    'optimizer': g_optimizer.state_dict()
                    }

                torch.save(state, os.path.join(dir_checkpoint, 'best_model.pth.tar'))

        log.update(epoch, train_log, val_log)

        
if __name__ == '__main__':
    args = parse_args()
    save_path = os.path.join(args.outf, args.save)
    if not os.path.exists(save_path):
        os.makedirs('%s' % save_path)
    with open(os.path.join(save_path,'params.txt'),'a') as f:
        f.write(str(args))
    log = Logger(save_path)
    #Set random seed for Pytorch and Numpy for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # model = UNet(n_channels=3, n_classes=2)
    import DR_segmentation as DRseg
    model = DRseg.Unet(encoder_name=args.backbone, encoder_weights=args.encoder_weights,lesion_classes = args.lesion_classes)
    model.cuda()

    g_optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=0.0005)
    resume = False
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            # checkpoint = torch.load(resume,map_location=lambda storage, loc: storage.cuda(int(args.gpu)))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']+1
            start_step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            g_optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model loaded from {}'.format(resume))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    else:
        start_epoch = 0
        start_step = 0


    # train_image_paths, train_mask_paths, train_vessel_paths = get_images_IDRiD(args.img_dir_train, args.preprocess, phase='train')
    # eval_image_paths, eval_mask_paths, eval_vessel_paths = get_images_IDRiD_test(args.img_dir_test, args.preprocess, phase='test')
    # train_dataset = multilesionDataset(train_image_paths, train_mask_paths, train_vessel_paths, transform=Compose([RandomRotation(args.rotation),]))
    # eval_dataset = multilesionDataset(eval_image_paths, eval_mask_paths, eval_vessel_paths)
    # train_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=True)
    # eval_loader = DataLoader(eval_dataset, args.eval_batch_size, shuffle=False)
    train_set, val_set = get_TrainDataset(args)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    g_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=150, gamma=0.1)
    criterion5 = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.crossentropy_weights5).to(device))
    criterion2 = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.crossentropy_weights2).to(device))

    train_model(model, train_loader, val_loader, criterion2, criterion5, g_optimizer, g_scheduler,  num_epochs=args.epochs, start_epoch=start_epoch, start_step=start_step, )
