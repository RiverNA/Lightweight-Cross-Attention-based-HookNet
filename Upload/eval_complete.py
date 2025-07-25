import torch
import torch.nn.functional as F
import os
import torchmetrics
import logging
import glob
import sys
import cv2
import sklearn.metrics
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from config import cfg
from dice_loss import dice_coeff
from loss import make_one_hot


def whole_preprocess(pil_img):
    img_nd = np.array(pil_img)
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    img_nd = img_nd.transpose((2, 0, 1))
    C, H, W = img_nd.shape

    mask = np.zeros([H, W])
    # background = np.where(img_nd == [0, 0, 0])
    main_text = np.where(img_nd[0, :, :] == 255)
    comment = np.where(img_nd[1, :, :] == 255)
    decoration = np.where(img_nd[2, :, :] == 255)

    # mask[background] = 0
    mask[main_text] = 1
    mask[comment] = 2
    mask[decoration] = 3

    return mask


def wholegt_preprocess(pil_img):
    img_nd = np.array(pil_img)
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    img_nd = img_nd.transpose((2, 0, 1))
    C, H, W = img_nd.shape
    mask = np.zeros([H, W])
    # background = np.where(img_nd == [0, 0, 0])
    main_text = np.where(img_nd[0, :, :] == 255)
    comment = np.where(img_nd[1, :, :] == 255)
    decoration = np.where(img_nd[2, :, :] == 255)
    # mask[background] = 0
    mask[main_text] = 1
    mask[comment] = 2
    mask[decoration] = 3

    return mask


def mask_to_image(pre_mask, save_path, suffix):
    c, h, w = pre_mask.shape
    out1 = np.zeros((c, h, w), dtype=np.uint8)
    out2 = np.zeros((c, h, w), dtype=np.uint8)
    out3 = np.zeros((c, h, w), dtype=np.uint8)

    # background = np.where(pre_mask.to('cpu') == 0)
    main_text = np.where(pre_mask.to('cpu') == 1)
    comment = np.where(pre_mask.to('cpu') == 2)
    decoration = np.where(pre_mask.to('cpu') == 3)
    out1[main_text] = 255
    out2[comment] = 255
    out3[decoration] = 255
    out = np.concatenate((out1, out2, out3), axis=0)
    out = out.transpose((1, 2, 0))
    cv2.imwrite(os.path.join(save_path, suffix + '.png'), out, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


def eval_net(model, loader, device):
    model.eval()
    mask_type = torch.int64
    n_val = len(loader)
    iou_ratio = 0
    test_save = os.path.join(os.environ['TMPDIR'], 'patch_save')
    if not os.path.exists(test_save):
        os.makedirs(test_save)
    with tqdm(total=n_val, desc='Round', unit='img') as pbar:
        for batch in loader:
            imgs_target = batch['image_target']
            imgs_context = batch['image_context']
            true_masks_target = batch['mask_target']
            suffix = batch['suffix'][0]

            imgs_target = imgs_target.to(device=device, dtype=torch.float32)
            imgs_context = imgs_context.to(device=device, dtype=torch.float32)
            true_masks_target = true_masks_target.to(device=device, dtype=mask_type)

            with torch.no_grad():
                masks_pred = model(imgs_context, imgs_target)

            if model.n_classes > 2:
                IOU = torchmetrics.JaccardIndex(task='multiclass', num_classes=model.n_classes, average='none')
                prd_target = F.log_softmax(masks_pred[1], dim=1)
                prd_target = torch.argmax(prd_target, dim=1)
                mask_to_image(prd_target, test_save, suffix)
                iou = IOU(prd_target.cpu().detach(), true_masks_target.cpu().detach())
                iou_ratio += iou
            else:
                IOU = torchmetrics.IoU(num_classes=2, absent_score=1)
                pred = torch.sigmoid(masks_pred[1])
                pred = (pred > 0.5).float()
                iou = IOU(pred.cpu().detach(), true_masks_target.type(torch.int64).cpu().detach())
                iou_ratio += iou
            pbar.update()

    piou_ratio = iou_ratio / n_val
    piou_ratio = sum(piou_ratio) / len(piou_ratio)

    groundtruth = os.path.join(os.environ['TMPDIR'], 'ori')
    whole_save = os.path.join(os.environ['TMPDIR'], 'whole_save')

    masks_ground = sorted(glob.glob(os.path.join(groundtruth, '*.png')))

    totensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    if not os.path.exists(whole_save):
        os.makedirs(whole_save)

    IOU = torchmetrics.JaccardIndex(task='multiclass', num_classes=model.n_classes, average='none')
    iou_ratio = 0
    for i in range(len(masks_ground)):
        ground = masks_ground[i]
        suffix = ground.split('/')[-1].split('.')[0]
        img = Image.open(ground).convert("RGB")
        W, H = img.size
        HH = H // 288 + 1
        WW = W // 288 + 1
        length = 288
        test = sorted(glob.glob(os.path.join(test_save, suffix + '*.png')))
        all = []
        for j in range(len(test)):
            ti = Image.open(test[j]).convert("RGB")
            all.append(ti)

        whole = Image.new('RGB', (WW * length, HH * length))

        for k in range(len(all)):
            whole.paste(all[k],
                        (length * (k % WW), length * (k // WW), length * (k % WW + 1), length * (k // WW + 1)))

        whole = transforms.CenterCrop((H, W))(whole)

        wholes = totensor(whole)
        save_image(wholes, os.path.join(whole_save, suffix + '.png'))

        img = wholegt_preprocess(img)
        whole = whole_preprocess(whole)

        iou = IOU(torch.from_numpy(whole).type(torch.int64), torch.from_numpy(img).type(torch.int64))
        iou_ratio += iou

    iou_ratio = iou_ratio / len(masks_ground)
    ave_iou = sum(iou_ratio) / len(iou_ratio)

    return piou_ratio, ave_iou, iou_ratio
