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

import transforms
from crowd_dataset import CrowdDataset

model_path = 'saved_model_path'

train_transform = torchvision.transforms.Compose([transforms.Rescale([564 // 8, 720 // 8]),
                                                  transforms.NegativeOneToOneNormalizeImage(),
                                                  transforms.NumpyArraysToTorchTensors()])

test_dataset = CrowdDataset('../storage/data/world_expo_datasets', 'test', transform=train_transform)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


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
for example_index, examples in enumerate(test_dataset_loader):
    images, labels, roi = examples
    images, labels, roi = Variable(images), Variable(labels), Variable(roi)
    predicted_density_maps, predicted_count_maps = net(images)
    predicted_density_maps, predicted_count_maps = predicted_density_maps.squeeze(dim=1), predicted_count_maps.squeeze(
        dim=1)
    predicted_density_maps = predicted_density_maps * roi
    predicted_count_maps = predicted_count_maps * roi
    density_loss = torch.abs(predicted_density_maps - labels).sum(1).sum(1).mean()
    count_loss = torch.abs(predicted_count_maps.sum(1).sum(1) - labels.sum(1).sum(1)).mean()
    running_count += labels.sum(1).sum(1).squeeze()
    running_count_error += count_loss.data[0]
    running_density_error += density_loss.data[0]
    if ((example_index + 1) % 120) == 0:
        print('Scene {}'.format(scene_number))
        print('Total count: {}'.format(running_count.data[0]))
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
