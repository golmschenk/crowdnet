"""
Code for a test session.
"""
import csv

import os

import numpy as np
from torch.autograd import Variable
from torch.nn import Module, Conv2d
from torch.nn.functional import leaky_relu
import torch.utils.data
import torchvision
import scipy.misc

import transforms
from crowd_dataset import CrowdDataset

model_path = 'saved_model_path'

patch_transform = transforms.ExtractPatchForPositionAndRescale()
test_transform = torchvision.transforms.Compose([transforms.NegativeOneToOneNormalizeImage(),
                                                 transforms.NumpyArraysToTorchTensors()])

test_dataset = CrowdDataset('../storage/data/world_expo_datasets', 'test')


class JointCNN(Module):
    """
    Basic CNN that produces only a density map.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(self.conv1.out_channels, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(self.conv2.out_channels, 128, kernel_size=3, padding=1)
        self.conv4 = Conv2d(self.conv3.out_channels, 256, kernel_size=3, padding=1)
        self.conv5 = Conv2d(self.conv4.out_channels, 10, kernel_size=3, padding=1)
        self.count_conv = Conv2d(self.conv5.out_channels, 1, kernel_size=1)
        self.density_conv = Conv2d(self.conv5.out_channels, 1, kernel_size=1)

    def __call__(self, *args, **kwargs):
        """
        Defined in subclass just to allow for type hinting.

        :return: The predicted labels.
        :rtype: torch.autograd.Variable
        """
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        """
        The forward pass of the network.

        :param x: The input images.
        :type x: torch.autograd.Variable
        :return: The predicted density labels.
        :rtype: torch.autograd.Variable
        """
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = leaky_relu(self.conv3(x))
        x = leaky_relu(self.conv4(x))
        x = leaky_relu(self.conv5(x))
        x_count = leaky_relu(self.count_conv(x))
        x_density = leaky_relu(self.density_conv(x))
        return x_density, x_count


net = JointCNN()
net.load_state_dict(torch.load(model_path))
count_errors = []
density_errors = []
print('Starting test...')
scene_number = 1
running_count = 0
running_count_error = 0
running_density_error = 0
for example_index, full_example in enumerate(test_dataset):
    number_of_pixels = full_example.shape[0] * full_example.shape[1]
    predicted_labels_stack = np.full([number_of_pixels, *full_example.labels.shape], fill_value=np.nan)
    full_predicted_count = 0
    pixel_index = 0
    for y in range(full_example.shape[0]):
        for x in range(full_example.shape[1]):
            example_patch, original_patch_size = patch_transform(full_example, y, x)
            example = test_transform(example_patch)
            image, label = Variable(example.image), Variable(example.label)
            predicted_label, predicted_count = net(image)
            predicted_label, predicted_count = predicted_label.numpy(), predicted_count.numpy()
            predicted_label_sum = np.sum(predicted_label)
            half_patch_size = int(original_patch_size // 2)
            predicted_label = scipy.misc.imresize(predicted_label, (2 * half_patch_size) + 1, mode='F')
            if predicted_label_sum != 0:
                unnormalized_predicted_label_sum = np.sum(predicted_label)
                predicted_label = (predicted_label / unnormalized_predicted_label_sum) * predicted_label_sum
            y_start_offset = 0
            if y - half_patch_size < 0:
                y_start_offset = half_patch_size - y
            y_end_offset = 0
            if y + half_patch_size > full_example.label.shape[0]:
                y_end_offset = y + half_patch_size - example.label.shape[0]
            x_start_offset = 0
            if x - half_patch_size < 0:
                x_start_offset = half_patch_size - x
            x_end_offset = 0
            if x + half_patch_size > full_example.label.shape[1]:
                x_end_offset = x + half_patch_size - example.label.shape[1]
            predicted_labels_stack[pixel_index,
                                   y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                   x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                   ] = predicted_label[y_start_offset:-y_end_offset,
                                                       x_start_offset:-x_end_offset]
            pixel_index += 1
            full_predicted_count += predicted_count / (((2 * half_patch_size) + 1) ** 2)
    full_predicted_label = np.nanmean(predicted_labels_stack, axis=0)
    density_loss = np.abs(full_predicted_label - full_example.label.numpy()).sum()
    count_loss = np.abs(full_predicted_count - full_example.label.numpy().sum())
    running_count += full_example.label.numpy().sum()
    running_count_error += count_loss
    running_density_error += density_loss
    if ((example_index + 1) % 120) == 0:
        print('Scene {}'.format(scene_number))
        print('Total count: {}'.format(running_count))
        count_error = running_count_error / 120
        print('Mean count error: {}'.format(count_error))
        density_error = running_density_error / 120
        print('Mean density error: {}'.format(density_error))
        count_errors.append(count_error)
        density_errors.append(density_error)
        running_count = 0
        running_count_error = 0
        running_density_error = 0
        scene_number += 1

csv_file_path = '../storage/logs/Test Results.csv'
if not os.path.isfile(csv_file_path):
    with open(csv_file_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Run Name', 'Scene 1', 'Scene 2', 'Scene 3', 'Scene 4', 'Scene 5', 'Mean',
                         'Scene 1 Density', 'Scene 2 Density', 'Scene 3 Density', 'Scene 4 Density', 'Scene 5 Density',
                         'Mean Density'])
with open(csv_file_path, 'a') as csv_file:
    writer = csv.writer(csv_file)
    test_results = [os.path.basename(model_path), *count_errors, np.mean(count_errors),
                    *density_errors, np.mean(density_errors)]
    writer.writerow(test_results)

print('Finished test.')
