from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import logging
import sys
import cv2
import numpy as np
from config import cfg
from torch import optim
from PIL import Image
from tqdm import tqdm
from eval_complete import eval_net
from Light_HookNet_model import Light_HookNet
from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from valid_dataset import BasDataset
from torch.utils.data import DataLoader
from loss import DiceLoss, make_one_hot

dir_checkpoint = cfg.output_path

scratch_train = os.path.join(os.environ['TMPDIR'], 'Training')
scratch_valid = os.path.join(os.environ['TMPDIR'], 'Validation')

dir_img_target = os.path.join(scratch_train, 'target_images')
dir_img_context = os.path.join(scratch_train, 'context_images')
dir_mask_target = os.path.join(scratch_train, 'target_masks')
dir_mask_context = os.path.join(scratch_train, 'context_masks')

valid_img_target = os.path.join(scratch_valid, 'target_images')
valid_img_context = os.path.join(scratch_valid, 'context_images')
valid_mask_target = os.path.join(scratch_valid, 'target_masks')
valid_mask_context = os.path.join(scratch_valid, 'context_masks')


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train_net(model,
              device,
              optimizer,
              scheduler,
              eepoch,
              epochs=5,
              batch_size=128,
              lr=0.001,
              save_cp=True,
              img_scale=1,
              classes=4,
              ):
    dataset = BasicDataset(dir_img_target, dir_img_context, dir_mask_target, dir_mask_context, img_scale)
    valid_dataset = BasDataset(valid_img_target, valid_img_context, valid_mask_target, valid_mask_context, img_scale)

    n_dataset = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True,
                              drop_last=False)

    logging.info(f'''Starting training:
        Total epochs:          {epochs}
        Batch size:            {batch_size}
        Initial learning rate: {lr}
        Save checkpoints:      {save_cp}
        Device:                {device.type}
    ''')

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = DiceLoss()

    for epoch in range(eepoch, epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_dataset, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs_target = batch['image_target']
                imgs_context = batch['image_context']
                true_masks_target = batch['mask_target']
                true_masks_context = batch['mask_context']
                assert imgs_target.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {imgs_target.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                masks = [true_masks_target]

                for i in range(4):
                    big_mask = masks[-1]
                    small_mask = F.avg_pool2d(big_mask, 2)
                    masks.append(small_mask)

                small = masks[1:]

                imgs_target = imgs_target.to(device=device, dtype=torch.float32)
                imgs_context = imgs_context.to(device=device, dtype=torch.float32)

                mask_type = torch.float32 if model.n_classes <= 2 else torch.int64
                true_masks_target = true_masks_target.to(device=device, dtype=mask_type)
                true_masks_context = true_masks_context.to(device=device, dtype=mask_type)
                masks_pred = model(imgs_context, imgs_target)

                true_masks_context_dice = torch.unsqueeze(true_masks_context, dim=1)
                true_masks_context_dice = make_one_hot(true_masks_context_dice.type(torch.int64), classes)
                true_masks_target_dice = torch.unsqueeze(true_masks_target, dim=1)
                true_masks_target_dice = make_one_hot(true_masks_target_dice.type(torch.int64), classes)

                c_loss1 = criterion1(masks_pred[0], true_masks_context)
                t_loss1 = criterion1(masks_pred[1], true_masks_target)

                c_loss2 = criterion2(masks_pred[0], true_masks_context_dice)
                t_loss2 = criterion2(masks_pred[1], true_masks_target_dice)

                c_loss = c_loss1 + c_loss2
                t_loss = t_loss1 + t_loss2

                loss_deep = 0

                for i in range(len(masks_pred[2])):
                    lossc = criterion1(masks_pred[2][2 - i], small[i].cuda().type(torch.int64))
                    dice = torch.unsqueeze(small[i].cuda().type(torch.int64), dim=1)
                    dice = make_one_hot(dice.type(torch.int64), classes)
                    lossd = criterion2(masks_pred[2][2 - i], dice)
                    loss_deep += lossc + lossd

                loss = t_loss + c_loss + 0.5 * loss_deep

                epoch_loss += loss.item() / imgs_context.shape[0]

                pbar.set_postfix({'loss': loss.item() / imgs_context.shape[0]})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(imgs_context.shape[0])

        print('Train average loss:', epoch_loss / n_dataset)
        print('lr:', optimizer.param_groups[0]['lr'])
        
        piou_ratio, ave_iou, iou_ratio = eval_net(model, valid_loader, device)
        print('Validation iou ratio: {}, {}'.format(ave_iou, iou_ratio))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                print('Created checkpoint directory')
            except OSError:
                pass
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'lr_schedule': scheduler.state_dict(),
                        },
                       os.path.join(dir_checkpoint, 'Light_HookNet_{:03d}.pth'.format(epoch + 1)))

            print('Checkpoint {:03d} saved at {}\n'.format((epoch + 1), os.path.join(dir_checkpoint,
                                                                                     'Light_HookNet_{:03d}.pth'.format(
                                                                                         epoch + 1))))

        if epoch < epochs - 1:
            scheduler.step()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    seed_torch(cfg.seed)

    cfg.n_classes = 4
    cfg.in_channels = 3
    cfg.n_filters = 16
    cfg.batch_size = 16
    cfg.total_epoch = 3
    eepoch = 0
    cfg.load = None
    model = Light_HookNet(cfg.in_channels, cfg.n_classes, cfg.filter_size, cfg.n_filters)
    model.init_weights()
    model.to(device=device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    if cfg.load:
        checkpoints = torch.load(cfg.load, map_location=device)
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        eepoch = checkpoints['epoch']
        scheduler.load_state_dict(checkpoints['lr_schedule'])
        scheduler.last_epoch = eepoch
        scheduler.step()
        print(f'Model loaded from {cfg.load}')

    try:
        train_net(model=model,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  eepoch=eepoch,
                  epochs=cfg.total_epoch,
                  batch_size=cfg.batch_size,
                  lr=cfg.learning_rate,
                  device=device,
                  img_scale=cfg.scale,
                  classes=cfg.n_classes,
                  )

    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
