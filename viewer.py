"""
Code for assisting in viewing the results.
"""

import matplotlib.cm
import numpy as np
import torch
import torchvision.utils


def convert_density_maps_to_heatmaps(label, predicted_label):
    """
    Converts a label and predicted label density map into their respective heatmap images.

    :param label: The label tensor.
    :type label: torch.autograd.Variable
    :param predicted_label: The predicted labels tensor.
    :type predicted_label: torch.autograd.Variable
    :return: The heatmap label tensor and heatmap predicted label tensor.
    :rtype: (torch.autograd.Variable, torch.autograd.Variable)
    """
    mappable = matplotlib.cm.ScalarMappable(cmap='inferno')
    label_array = label.numpy()
    predicted_label_array = predicted_label.numpy()
    mappable.set_clim(vmin=min(label_array.min(), predicted_label_array.min()),
                      vmax=max(label_array.max(), predicted_label_array.max()))
    label_heatmap_array = mappable.to_rgba(label_array).astype(np.float32)
    label_heatmap_tensor = torch.from_numpy(label_heatmap_array[:, :, :3].transpose((2, 0, 1)))
    predicted_label_heatmap_array = mappable.to_rgba(predicted_label_array).astype(np.float32)
    predicted_label_heatmap_tensor = torch.from_numpy(predicted_label_heatmap_array[:, :, :3].transpose((2, 0, 1)))
    return label_heatmap_tensor, predicted_label_heatmap_tensor


def create_crowd_images_comparison_grid(images, labels, predicted_labels, number_of_images=3):
    """
    Creates a grid of images from the original images, the true labels, and the predicted labels.

    :param images: The original RGB images.
    :type images: torch.autograd.Variable
    :param labels: The labels.
    :type labels: torch.autograd.Variable
    :param predicted_labels: The predicted labels.
    :type predicted_labels: torch.autograd.Variable
    :param number_of_images: The number of (original) images to include in the grid.
    :type number_of_images: int
    :return: The image of the grid of images.
    :rtype: np.ndarray
    """
    grid_image_list = []
    for index in range(min(number_of_images, images.size()[0])):
        grid_image_list.append((images[index].data + 1) / 2)
        label_heatmap, predicted_label_heatmap = convert_density_maps_to_heatmaps(labels[index].data,
                                                                                  predicted_labels[index].data)
        grid_image_list.append(label_heatmap)
        grid_image_list.append(predicted_label_heatmap)
    return torchvision.utils.make_grid(grid_image_list, nrow=number_of_images)
