import numpy as np
import torch
from PIL import Image
from torchvision import transforms

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 1.0)
    return image_np


def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor


def random_image(base_img):
    shape = np.asarray(base_img).shape
    mode = base_img.mode

    new_array = np.random.random(shape)
    new_image = Image.fromarray(new_array, mode=mode)

    print(np.asarray(base_img).shape, np.asarray(new_image).shape)
    print(base_img.size, new_image.size)
    print(base_img.mode, new_image.mode)
    print(np.asarray(base_img), np.asarray(new_image))
    raise NotImplementedError

    return new_image
