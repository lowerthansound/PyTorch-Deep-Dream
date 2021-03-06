print("Import things") # Importing can take quite a while

import argparse
import os
import pathlib

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


def main():
    args = get_args()
    model = get_model(args.layer)

    print("Deep dream")
    for image_path in INPUT_DIR.iterdir():

        print(f'  {image_path}')

        # Load image from input
        image = Image.open(image_path)

        # Deep dream
        dreamed_image = deep_dream(
            image,
            model,
            num_octaves=args.num_octaves,
            octave_scale=args.octave_scale,
            iterations=args.iterations,
            lr=args.step_size,
        )

        # Save image to output
        output = OUTPUT_DIR / image_path.name
        plt.imsave(str(output), dreamed_image)


def get_args():

    print("Get arguments")

    parser = argparse.ArgumentParser()

    parser.add_argument("--layer", type=int, default=27, help="Layer to maximize output")
    parser.add_argument("--num-octaves", type=int, default=10, help="Number of octaves")
    parser.add_argument("--octave-scale", type=float, default=1.4, help="Image scale between octaves")
    parser.add_argument("--iterations", type=int, default=20, help="Number of gradient ascent steps performed for each octave")
    parser.add_argument("--step-size", type=float, default=0.01, help="Learning rate for the gradient ascent step")

    return parser.parse_args()


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
