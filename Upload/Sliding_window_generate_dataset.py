import glob
import os
import PIL
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

target_size = 288
interval = target_size * 2

ip = 'train_path/im_center_crop'
mp = 'train_path/ms_center_crop'

images = sorted(glob.glob(os.path.join(ip, '*')))
masks = sorted(glob.glob(os.path.join(mp, '*')))

save_path_context = 'train_path/Transform_images'
save_path_context_mask = 'train_path/Transform_masks'

totensor = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context, suffix + '_{:06d}.png'.format(id)))
                id += 1

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]
    toimage = totensor(mask)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:06d}'.format(id) + '_zones_NA.png'))
                id += 1

ipv = 'validation_path/im_center_crop'
mpv = 'validation_path/ms_center_crop'

imagesv = sorted(glob.glob(os.path.join(ipv, '*')))
masksv = sorted(glob.glob(os.path.join(mpv, '*')))

save_path_context = 'validation_path/Transform_images'
save_path_context_mask = 'validation_path/Transform_masks'

totensor = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(imagesv)):
    image = Image.open(imagesv[i])
    suffix = imagesv[i].split('/')[-1].split('.')[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context, suffix + '_{:06d}.png'.format(id)))
                id += 1

for i in range(len(masksv)):
    mask = Image.open(masksv[i])
    suffix = masksv[i].split('/')[-1].split('.')[0]
    toimage = totensor(mask)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:06d}'.format(id) + '_zones_NA.png'))
                id += 1

ipt = 'test_path/im_center_crop'
mpt = 'test_path/ms_center_crop'

imagest = sorted(glob.glob(os.path.join(ipt, '*')))
maskst = sorted(glob.glob(os.path.join(mpt, '*')))

save_path_context = 'test_path/Transform_images'
save_path_context_mask = 'test_path/Transform_masks'

totensor = transforms.Compose([
    transforms.ToTensor(),
])

if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(imagest)):
    image = Image.open(imagest[i])
    suffix = imagest[i].split('/')[-1].split('.')[0]
    toimage = totensor(image)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context, suffix + '_{:06d}.png'.format(id)))
                id += 1

for i in range(len(maskst)):
    mask = Image.open(maskst[i])
    suffix = maskst[i].split('/')[-1].split('.')[0]
    toimage = totensor(mask)
    _, h, w = toimage.shape
    id = 0
    for i in range(0, h + 1, target_size):
        for j in range(0, w + 1, target_size):
            if i + interval <= h and j + interval <= w:
                crop = toimage[:, i:i + interval, j:j + interval]
                save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:06d}'.format(id) + '_zones_NA.png'))
                id += 1
