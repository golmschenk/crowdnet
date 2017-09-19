"""
Code for a test session.
"""
import csv

import os

import numpy as np
from torch.autograd import Variable
from torch.nn import Module, Conv2d, MaxPool2d
from torch.nn.functional import relu
import torch.utils.data
import torchvision
import scipy.misc

import transforms
import settings
from crowd_dataset import CrowdDataset

patch_transform = transforms.ExtractPatchForPositionAndRescale()
test_transform = torchvision.transforms.Compose([transforms.NegativeOneToOneNormalizeImage(),
                                                 transforms.NumpyArraysToTorchTensors()])

test_dataset = CrowdDataset(settings.database_path, 'test')


class JointCNN(Module):
    """
    Basic CNN that produces only a density map.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=7, padding=3)
        self.max_pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(self.conv1.out_channels, 32, kernel_size=7, padding=3)
        self.max_pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv2d(self.conv2.out_channels, 64, kernel_size=5, padding=2)
        self.conv4 = Conv2d(self.conv3.out_channels, 1000, kernel_size=18)
        self.conv5 = Conv2d(self.conv4.out_channels, 400, kernel_size=1)
        self.count_conv = Conv2d(self.conv5.out_channels, 1, kernel_size=1)
        self.density_conv = Conv2d(self.conv5.out_channels, 324, kernel_size=1)

    def __call__(self, *args, **kwargs):
        """
        Defined in subclass just to allow for type hinting.

        :return: The predicted labels.
        :rtype: torch.autograd.Variable
        """
        return super().__call__(*args, **kwargs)

    def forward(self, z):
        """
        The forward pass of the network.

        :param z: The input images.
        :type z: torch.autograd.Variable
        :return: The predicted density labels.
        :rtype: torch.autograd.Variable
        """
        z = relu(self.conv1(z))
        z = self.max_pool1(z)
        z = relu(self.conv2(z))
        z = self.max_pool2(z)
        z = relu(self.conv3(z))
        z = relu(self.conv4(z))
        z = relu(self.conv5(z))
        z_count = self.count_conv(z).view(-1)
        z_density = self.density_conv(z).view(-1, 18, 18)
        return z_density, z_count


net = JointCNN()
net.load_state_dict(torch.load(settings.test_model_path))
count_errors = []
density_errors = []
print('Starting test...')
scene_number = 1
running_count = 0
running_count_error = 0
running_density_error = 0
for full_example_index, full_example in enumerate(test_dataset):
    print('Processing example {}\r'.format(full_example_index))
    bin_predicted_label = np.zeros_like(full_example.label, dtype=np.float32)
    hit_predicted_label = np.zeros_like(full_example.label, dtype=np.int32)
    full_predicted_count = 0
    for y in range(full_example.label.shape[0]):
        for x in range(full_example.label.shape[1]):
            example_patch, original_patch_size = patch_transform(full_example, y, x)
            example = test_transform(example_patch)
            image, label = Variable(example.image.unsqueeze(0)), Variable(example.label)
            predicted_label, predicted_count = net(image)
            predicted_label, predicted_count = predicted_label.data.squeeze(0).numpy(), predicted_count.data.squeeze(0).numpy()
            predicted_label_sum = np.sum(predicted_label)
            half_patch_size = int(original_patch_size // 2)
            original_patch_dimensions = ((2 * half_patch_size) + 1, (2 * half_patch_size) + 1)
            predicted_label = scipy.misc.imresize(predicted_label, original_patch_dimensions, mode='F')
            if predicted_label_sum != 0:
                unnormalized_predicted_label_sum = np.sum(predicted_label)
                predicted_label = (predicted_label / unnormalized_predicted_label_sum) * predicted_label_sum
            y_start_offset = 0
            if y - half_patch_size < 0:
                y_start_offset = half_patch_size - y
            y_end_offset = 0
            if y + half_patch_size >= full_example.label.shape[0]:
                y_end_offset = y + half_patch_size + 1 - full_example.label.shape[0]
            x_start_offset = 0
            if x - half_patch_size < 0:
                x_start_offset = half_patch_size - x
            x_end_offset = 0
            if x + half_patch_size >= full_example.label.shape[1]:
                x_end_offset = x + half_patch_size + 1 - full_example.label.shape[1]
                bin_predicted_label[
                                    y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                    x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                    ] += predicted_label[y_start_offset:predicted_label.shape[0] - y_end_offset,
                                                         x_start_offset:predicted_label.shape[1] - x_end_offset]
                hit_predicted_label[
                                    y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                    x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                    ] += 1
            full_predicted_count += predicted_count / (((2 * half_patch_size) + 1) ** 2)
    full_predicted_label = bin_predicted_label / hit_predicted_label.astype(np.float32)
    density_loss = np.abs(full_predicted_label - full_example.label.numpy()).sum()
    count_loss = np.abs(full_predicted_count - full_example.label.numpy().sum())
    running_count += full_example.label.numpy().sum()
    running_count_error += count_loss
    running_density_error += density_loss
    if ((full_example_index + 1) % 120) == 0:
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

csv_file_path = os.path.join(settings.log_directory, 'Test Results.csv')
if not os.path.isfile(csv_file_path):
    with open(csv_file_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Run Name', 'Scene 1', 'Scene 2', 'Scene 3', 'Scene 4', 'Scene 5', 'Mean',
                         'Scene 1 Density', 'Scene 2 Density', 'Scene 3 Density', 'Scene 4 Density', 'Scene 5 Density',
                         'Mean Density'])
with open(csv_file_path, 'a') as csv_file:
    writer = csv.writer(csv_file)
    test_results = [os.path.basename(settings.test_model_path), *count_errors, np.mean(count_errors),
                    *density_errors, np.mean(density_errors)]
    writer.writerow(test_results)

print('Finished test.')
