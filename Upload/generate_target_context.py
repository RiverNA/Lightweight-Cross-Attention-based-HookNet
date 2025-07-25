import glob
import os
import PIL
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms import InterpolationMode

size = 288

ti = 'train_path/Transform_images'
mi = 'train_path/Transform_masks'

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))

save_path_target = 'train_path/Training/target_images'
save_path_context = 'train_path/Training/context_images'
save_path_target_mask = 'train_path/Training/target_masks'
save_path_context_mask = 'train_path/Training/context_masks'

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST_EXACT),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))

ti = 'validation_path/Transform_images'
mi = 'validation_path/Transform_masks'

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))

save_path_target = 'validation_path/Validation/target_images'
save_path_context = 'validation_path/Validation/context_images'
save_path_target_mask = 'validation_path/Validation/target_masks'
save_path_context_mask = 'validation_path/Validation/context_masks'

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST_EXACT),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))

ti = 'test_path/Transform_images'
mi = 'test_path/Transform_masks'

images = sorted(glob.glob(os.path.join(ti, '*.png')))
masks = sorted(glob.glob(os.path.join(mi, '*.png')))

save_path_target = 'test_path/Testing/target_images'
save_path_context = 'test_path/Testing/context_images'
save_path_target_mask = 'test_path/Testing/target_masks'
save_path_context_mask = 'test_path/Testing/context_masks'

target = transforms.Compose([
    transforms.CenterCrop((size, size)),
    transforms.ToTensor(),
])

context_images = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

context_masks = transforms.Compose([
    transforms.CenterCrop((size * 2, size * 2)),
    transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST_EXACT),
    transforms.ToTensor(),
])

if not os.path.exists(save_path_target):
    os.makedirs(save_path_target)
if not os.path.exists(save_path_context):
    os.makedirs(save_path_context)
if not os.path.exists(save_path_target_mask):
    os.makedirs(save_path_target_mask)
if not os.path.exists(save_path_context_mask):
    os.makedirs(save_path_context_mask)

for i in range(len(images)):
    image = Image.open(images[i])
    suffix = images[i].split('/')[-1].split('.')[0]

    target_image = target(image)
    context_image = context_images(image)
    save_image(target_image, os.path.join(save_path_target, suffix + '.png'))
    save_image(context_image, os.path.join(save_path_context, suffix + '.png'))

for i in range(len(masks)):
    mask = Image.open(masks[i])
    suffix = masks[i].split('/')[-1].split('.')[0]

    target_mask = target(mask)
    context_mask = context_masks(mask)
    save_image(target_mask, os.path.join(save_path_target_mask, suffix + '.png'))
    save_image(context_mask, os.path.join(save_path_context_mask, suffix + '.png'))
