"""
Code for custom transforms.
"""
from collections import namedtuple

import torch
import scipy.misc
import numpy as np
import random

from pytorch_crowd_dataset import CrowdExample


class NumpyArrayToTorchTensor:
    """
    Converts from NumPy arrays of an example to Torch tensors.
    """
    def __call__(self, example):
        image, label, roi = example.image, example.label, example.roi
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        roi = torch.from_numpy(roi.astype(np.uint8))
        return CrowdExample(image=image, label=label, roi=roi)


class Rescale:
    """
    2D rescaling of an example (when in NumPy HWC form).
    """
    def __init__(self, scaled_size):
        self.scaled_size = scaled_size

    def __call__(self, example):
        image = scipy.misc.imresize(example.image, self.scaled_size)
        original_label_sum = np.sum(example.label)
        label = scipy.misc.imresize(example.label, self.scaled_size, mode='F')
        if original_label_sum != 0:
            unnormalized_label_sum = np.sum(label)
            label = (label / unnormalized_label_sum) * original_label_sum

        roi = scipy.misc.imresize(example.roi, self.scaled_size, mode='F') > 0.5
        return CrowdExample(image=image, label=label, roi=roi)


class RandomHorizontalFlip:
    """
    Randomly flips the example horizontally (when in NumPy HWC form).
    """
    def __call__(self, example):
        if random.choice([True, False]):
            image = np.flip(example.image, axis=1).copy()
            label = np.flip(example.label, axis=1).copy()
            roi = np.flip(example.roi, axis=1).copy()
            return CrowdExample(image=image, label=label, roi=roi)
        else:
            return example


class NormalizeImage:
    """
    Normalizes a uint8 image to range -1 to 1.
    """
    def __call__(self, example):
        image = (example.image.astype(np.float32) / (255 / 2)) - 1
        return CrowdExample(image=image, label=example.label, roi=example.roi)
