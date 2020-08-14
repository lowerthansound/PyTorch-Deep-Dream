print("Import things") # Importing can take quite a while

import argparse
import os
import pathlib
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
import scipy.ndimage as nd
from torchvision import models
from torch.autograd import Variable
from PIL import Image

from utils import deprocess, preprocess, clip


INPUT_DIR = pathlib.Path('input')
OUTPUT_DIR = pathlib.Path('output')

WARMUP_DIR = OUTPUT_DIR / 'warmup'
MURILO_DIR = OUTPUT_DIR / 'murilo'
RANDOM_DIR = OUTPUT_DIR / 'random'
TIMING_FILE = OUTPUT_DIR / 'timing.txt'


def main():
    model = get_model(layer=34)
    num_octaves = 10
    octave_scale = 1.35 # minimum that works for 240p 10 octaves
    iterations = 20
    step_size = 0.01

    WARMUP_DIR.mkdir(parents=True, exist_ok=True)
    MURILO_DIR.mkdir(parents=True, exist_ok=True)
    RANDOM_DIR.mkdir(parents=True, exist_ok=True)
    try:
        TIMING_FILE.unlink()
    except FileNotFoundError:
        pass

    print("Warming up with a separate image")
    for image_path in [INPUT_DIR / 'sky.jpeg']:
        print(f'  {image_path}')
        image = Image.open(image_path)
        t0 = time()
        dreamed_image = deep_dream(
            image,
            model,
            num_octaves=num_octaves,
            octave_scale=octave_scale,
            iterations=iterations,
            lr=step_size,
        )
        t1 = time()
        # Save image to output
        output = WARMUP_DIR / 'sky.png'
        plt.imsave(str(output), dreamed_image)
        # Save timing
        with TIMING_FILE.open(mode='a') as f:
            relative_name = str(output.relative_to(OUTPUT_DIR))
            time_spent = t1 - t0
            f.write(f"{relative_name: >20s}: {time_spent:.2f}s\n")

    print("Dreaming murilo images to check the effect and measure time")
    for image_path in INPUT_DIR.glob('murilo_*.png'):
        print(f'  {image_path}')
        image = Image.open(image_path)
        for i in range(3):
            print(f'    {i:02d}')
            t0 = time()
            dreamed_image = deep_dream(
                image,
                model,
                num_octaves=num_octaves,
                octave_scale=octave_scale,
                iterations=iterations,
                lr=step_size,
            )
            t1 = time()
            # Save image to output
            name = f"{image_path.with_suffix('').name}.{i:02d}.png"
            output = MURILO_DIR / name
            plt.imsave(str(output), dreamed_image)
            # Save timing
            with TIMING_FILE.open(mode='a') as f:
                relative_name = str(output.relative_to(OUTPUT_DIR))
                time_spent = t1 - t0
                f.write(f"{relative_name: >20s}: {time_spent:.2f}s\n")

    print("Dream random images to measure time")
    for image_path in INPUT_DIR.glob('murilo_*.png'):
        base_image = Image.open(image_path)
        width = base_image.width
        height = base_image.height
        print(f'  {height}p')
        for i in range(3):
            print(f'    {i:02d}')
            image = Image.effect_noise((width, height), 40)
            t0 = time()
            dreamed_image = deep_dream(
                image,
                model,
                num_octaves=num_octaves,
                octave_scale=octave_scale,
                iterations=iterations,
                lr=step_size,
            )
            t1 = time()
            # Save image to output
            name = f'{image.height}p_{i:02d}.png'
            output = RANDOM_DIR / name
            plt.imsave(str(output), dreamed_image)
            # Save timing
            with TIMING_FILE.open(mode='a') as f:
                relative_name = str(output.relative_to(OUTPUT_DIR))
                time_spent = t1 - t0
                f.write(f"{relative_name: >20s}: {time_spent:.2f}s\n")


def get_model(layer):

    # Layer indicates the layer we use for "optimizing" the image
    # We will use the model up until that layer only

    print("Get model")

    # Retrieve layers from template model
    template = models.vgg19(pretrained=True)
    children = list(template.features.children())

    # Create new (trimmed) model based on previous layers
    new_model = nn.Sequential(*children[:layer+1])
    if torch.cuda.is_available():
        new_model = new_model.cuda()

    return new_model


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


if __name__ == "__main__":
    main()
