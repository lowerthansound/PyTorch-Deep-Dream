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


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", default=20, help="number of gradient ascent steps per octave")
    parser.add_argument("--at_layer", default=27, type=int, help="layer at which we modify image to maximize outputs")
    parser.add_argument("--lr", default=0.01, help="learning rate")
    parser.add_argument("--octave_scale", default=1.4, help="image scale between octaves")
    parser.add_argument("--num_octaves", default=10, help="number of octaves")
    args = parser.parse_args()

    # Define the model
    network = models.vgg19(pretrained=True)
    layers = list(network.features.children())
    model = nn.Sequential(*layers[: (args.at_layer + 1)])
    if torch.cuda.is_available():
        model = model.cuda()

    # Deep dream images
    for image_path in INPUT_DIR.iter():

        # Load image from input
        image = Image.open(image_path)

        # Deep dream
        dreamed_image = deep_dream(
            image,
            model,
            iterations=args.iterations,
            lr=args.lr,
            octave_scale=args.octave_scale,
            num_octaves=args.num_octaves,
        )

        # Save image to output
        output = OUTPUT_DIR / image_path.filename
        plt.imsave(str(output), dreamed_image)


if __name__ == "__main__":
    main()
