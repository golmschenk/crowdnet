"""
Code for custom transforms.
"""

import torch
import scipy.misc
import numpy as np
import random


class NumpyArrayToTorchTensor:
    """
    Converts from NumPy arrays of an example to Torch tensors.
    """
    def __call__(self, example):
        image, label, roi = example.image, example.label, example.roi
        image = image.transpose((2, 0, 1))
        example.image = torch.from_numpy(image)
        example.label = torch.from_numpy(label)
        example.roi = torch.from_numpy(roi)
        return example


class Rescale:
    """
    2D rescaling of an example (when in NumPy HWC form).
    """
    def __init__(self, scaled_size):
        self.scaled_size = scaled_size

    def __call__(self, example):
        example.image = scipy.misc.imresize(example.image, self.scaled_size, mode='F')
        original_label_sum = np.sum(example.label)
        label = scipy.misc.imresize(example.label, self.scaled_size, mode='F')
        if original_label_sum != 0:
            unnormalized_label_sum = np.sum(label)
            label = (label / unnormalized_label_sum) * original_label_sum
        example.label = label
        example.roi = scipy.misc.imresize(example.roi, self.scaled_size, mode='F') > 0.5
        return example


class RandomHorizontalFlip:
    """
    Randomly flips the example horizontally (when in NumPy HWC form).
    """
    def __call__(self, example):
        if random.choice([True, False]):
            example = np.flip(example, axis=1)
        return example
