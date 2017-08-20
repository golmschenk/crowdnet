"""
Code for assisting in viewing the results.
"""

import matplotlib.cm
import numpy as np
import torch
import torchvision.utils


def convert_single_channel_to_heatmap(single_channel_image):
    """
    Converts a single channel image to a heatmap image.

    :param single_channel_image: The single channel image.
    :type single_channel_image: np.ndarray
    :return: The heatmap image.
    :rtype: np.ndarray
    """
    colormap = matplotlib.cm.ScalarMappable(cmap='inferno')
    heatmap_image = colormap.to_rgba(single_channel_image).astype(np.float32)
    return heatmap_image


def convert_single_channel_tensor_to_heatmap_tensor(single_channel_tensor):
    heatmap_array = convert_single_channel_to_heatmap(single_channel_tensor.numpy())
    return torch.from_numpy(heatmap_array[:, :, :3].transpose((2, 0, 1)))


def create_crowd_images_comparison_grid(images, labels, predicted_labels, number_of_images=3):
    grid_image_list = []
    for index in range(min(number_of_images, images.size()[0])):
        grid_image_list.append((images[index].data + 1) / 2)
        grid_image_list.append(convert_single_channel_tensor_to_heatmap_tensor(labels[index].data))
        grid_image_list.append(convert_single_channel_tensor_to_heatmap_tensor(predicted_labels[index].data))
    return torchvision.utils.make_grid(grid_image_list, nrow=number_of_images)
