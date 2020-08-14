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

ERROR_FILE = OUTPUT_DIR / 'errors.txt'
TIMINGS_FILE = OUTPUT_DIR / 'timings.txt'


def main():

    # Load base model
    print("Load base model")
    base_model = models.vgg19(pretrained=True)
    base_layers = list(base_model.features.children())
    nlayers = len(base_layers)
    print(f"This model has {nlayers} layers")

    # For each layer
    for layer in range(nlayers):

        # Create model
        try:
            print(f"Creating model up until layer {layer}")
            model = nn.Sequential(*base_layers[0:layer+1])
            if torch.cuda.is_available():
                model = model.cuda()
        except Exception as e:
            msg = f"Error to create layer {layer:04d}: {e}"
            print(msg)
            with ERROR_FILE.open(mode='a') as f:
                f.write(msg + '\n')
            continue

        # For each input
        for image_path in INPUT_DIR.glob('*'):
            try:
                # Load image
                image = Image.open(str(image_path))

                # Deep dream image
                t0 = time()
                dreamed_image = deep_dream(
                    image,
                    model,
                    num_octaves=10,
                    octave_scale=1.4,
                    iterations=10,
                    lr=0.02,
                )
                t1 = time()

                # Save output to output/layer-<LAYER>/<IMAGE>.png
                output_path = str(OUTPUT_DIR / f'layer-{layer:04d}' / image_path.name)
                plt.imsave(output_path, dreamed_image)

                # Save timings to output/timings.txt
                elapsed_time = t1 - t0
                with TIMINGS_FILE.open(mode='a') as f:
                    benchmark_line = f"{output_path: >45s}: {elapsed_time:.1f}s\n"
                    f.write(benchmark_line)
            except Exception as e:
                msg = f"Error to process {image_path.name} at layer {layer:04d}: {e}"
                print(msg)
                with ERROR_FILE.open(mode='a') as f:
                    f.write(msg + '\n')


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
