"""
Code for custom transforms.
"""
from collections import namedtuple

import torch
import scipy.misc
import numpy as np
import random

from crowd_dataset import CrowdExample, CrowdExampleWithPerspective


class NumpyArraysToTorchTensors:
    """
    Converts from NumPy arrays of an example to Torch tensors.
    """

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy.
        :type example: CrowdExample
        :return: The crowd example in Tensors.
        :rtype: CrowdExample
        """
        image, label, roi = example.image, example.label, example.roi
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        roi = torch.from_numpy(roi.astype(np.float32))
        return CrowdExample(image=image, label=label, roi=roi)


class Rescale:
    """
    2D rescaling of an example (when in NumPy HWC form).
    """

    def __init__(self, scaled_size):
        self.scaled_size = scaled_size

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy.
        :type example: CrowdExample
        :return: The crowd example in Numpy with each of the arrays resized.
        :rtype: CrowdExample
        """
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
        """
        :param example: A crowd example in NumPy.
        :type example: CrowdExample
        :return: The possibly flipped crowd example in Numpy.
        :rtype: CrowdExample
        """
        if random.choice([True, False]):
            image = np.flip(example.image, axis=1).copy()
            label = np.flip(example.label, axis=1).copy()
            roi = np.flip(example.roi, axis=1).copy()
            return CrowdExample(image=image, label=label, roi=roi)
        else:
            return example


class NegativeOneToOneNormalizeImage:
    """
    Normalizes a uint8 image to range -1 to 1.
    """

    def __call__(self, example):
        """
        :param example: A crowd example in NumPy with image from 0 to 255.
        :type example: CrowdExample
        :return: A crowd example in NumPy with image from -1 to 1.
        :rtype: CrowdExample
        """
        image = (example.image.astype(np.float32) / (255 / 2)) - 1
        return CrowdExample(image=image, label=example.label, roi=example.roi)


class RandomlySelectPatchAndRescale:
    """
    Selects a patch of the example and resizes it based on the perspective map.
    """

    def __init__(self, scaled_size):
        self.scaled_size = scaled_size

    def __call__(self, example_with_perspective):
        """
        :param example_with_perspective: A crowd example with perspective.
        :type example_with_perspective: CrowdExampleWithPerspective
        :return: A crowd example.
        :rtype: CrowdExample
        """
        while True:
            y, x = self.select_random_position(example_with_perspective)
            patch = self.get_patch_for_position(y, x, example_with_perspective)
            if np.any(patch.roi):
                example = self.resize_patch(patch)
                return example

    @staticmethod
    def select_random_position(example_with_perspective):
        """
        Picks a random position in the full example.

        :param example_with_perspective: The full example with perspective.
        :type example_with_perspective: CrowdExampleWithPerspective
        :return: The y and x positions chosen randomly.
        :rtype: (int, int)
        """
        y = np.random.randint(example_with_perspective.label.shape[0])
        x = np.random.randint(example_with_perspective.label.shape[1])
        return y, x

    def get_patch_for_position(self, y, x, example_with_perspective):
        """
        Retrieves the patch for a given position.

        :param y: The y center of the patch.
        :type y: int
        :param x: The x center of the patch.
        :type x: int
        :param example_with_perspective: The full example with perspective to extract the patch from.
        :type example_with_perspective: CrowdExampleWithPerspective
        :return: The patch.
        :rtype: CrowdExample
        """
        pixels_per_meter = example_with_perspective.perspective[y, x]
        example = CrowdExample(image=example_with_perspective.image, label=example_with_perspective.label,
                               roi=example_with_perspective.roi)
        patch_size = 3 * pixels_per_meter
        half_patch_size = patch_size // 2
        if y - half_patch_size < 0:
            example = self.pad_example(example, y_padding=(half_patch_size - y, 0))
            y += half_patch_size - y
        if y + half_patch_size > example.label.shape[0]:
            example = self.pad_example(example, y_padding=(0, y + half_patch_size - example.label.shape[0]))
        if x - half_patch_size < 0:
            example = self.pad_example(example, x_padding=(half_patch_size - x, 0))
            x += half_patch_size - x
        if x + half_patch_size > example.label.shape[1]:
            example = self.pad_example(example, x_padding=(0, x + half_patch_size - example.label.shape[1]))
        image_patch = example.image[y - half_patch_size:y + half_patch_size + 1,
                                    x - half_patch_size:x + half_patch_size + 1,
                                    :]
        label_patch = example.label[y - half_patch_size:y + half_patch_size + 1,
                                    x - half_patch_size:x + half_patch_size + 1]
        roi_patch = example.roi[y - half_patch_size:y + half_patch_size + 1,
                                x - half_patch_size:x + half_patch_size + 1]
        return CrowdExample(image=image_patch, label=label_patch, roi=roi_patch)

    @staticmethod
    def pad_example(example, y_padding=(0, 0), x_padding=(0, 0)):
        """
        Pads the example.

        :param example: The example to pad.
        :type example: CrowdExample
        :param y_padding: The amount to pad the y dimension.
        :type y_padding: (int, int)
        :param x_padding: The amount to pad the x dimension.
        :type x_padding: (int, int)
        :return: The padded example.
        :rtype: CrowdExample
        """
        z_padding = (0, 0)
        image = np.pad(example.image, (y_padding, x_padding, z_padding), 'constant')
        label = np.pad(example.label, (y_padding, x_padding), 'constant')
        roi = np.pad(example.roi, (y_padding, x_padding), 'constant', constant_values=False)
        return CrowdExample(image=image, label=label, roi=roi)

    def resize_patch(self, patch):
        """
        :param patch: The patch to resize.
        :type patch: CrowdExample
        :return: The crowd example that is the resized patch.
        :rtype: CrowdExample
        """
        image = scipy.misc.imresize(patch.image, self.scaled_size)
        original_label_sum = np.sum(patch.label)
        label = scipy.misc.imresize(patch.label, self.scaled_size, mode='F')
        if original_label_sum != 0:
            unnormalized_label_sum = np.sum(label)
            label = (label / unnormalized_label_sum) * original_label_sum
        roi = scipy.misc.imresize(patch.roi, self.scaled_size, mode='F') > 0.5
        return CrowdExample(image=image, label=label, roi=roi)
