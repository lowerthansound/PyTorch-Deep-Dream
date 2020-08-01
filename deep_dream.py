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


def main():

    print("Load image input")
    image = Image.open(INPUT_DIR / 'sky.jpeg')


    print("Load base model")
    base_model = models.vgg19(pretrained=True)
    base_layers = list(base_model.features.children())


    nlayers = len(base_layers)
    print(f"This model has {nlayers} layers")
    for layer in range(nlayers):

        try:

            print("Creating model up until layer {layer}")

            t0 = time()

            new_model = nn.Sequential(*base_layers[0:layer+1])
            if torch.cuda.is_available():
                new_model = new_model.cuda()

            print("Dream")

            t1 = time()

            # Deep dream image
            dreamed_image = deep_dream(
                image,
                model,
                num_octaves=10,
                octave_scale=1.4,
                iterations=10,
                lr=0.1,
            )

            t2 = time()

            print("Saving output and timings")

            # Save image to output
            name = f"output_layer-{layer:04d}.jpeg"
            plt.imsave(str(OUTPUT_DIR / name), dreamed_image)

            # Save timings
            dreaming_time = t2 - t1
            with (OUTPUT_DIR / 'timings.txt').open(mode='a') as f:
                f.write(f"{name}: {dreaming_time:.1f}s\n")

        except Exception as e:

            print(f"Error at layer {layer}: {e}")
            with (OUTPUT_DIR / 'errors.txt').open(mode='a') as f:
                f.write(f"Error at layer {layer:04d}: {e}\n")


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
