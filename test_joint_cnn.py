"""
Main code for a training session.
"""
import os
import datetime
from torch.autograd import Variable
from torch.nn import Module, Conv2d, L1Loss
from torch.nn.functional import leaky_relu
from torch.optim import Adam
import torch.utils.data
import torchvision
from tensorboard import SummaryWriter

import transforms
import viewer
from crowd_dataset import CrowdDataset

run_name = 'Joint CNN'

train_transform = torchvision.transforms.Compose([transforms.Rescale([564 // 8, 720 // 8]),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.NegativeOneToOneNormalizeImage(),
                                                  transforms.NumpyArraysToTorchTensors()])

test_dataset = CrowdDataset('../storage/data/world_expo_datasets', 'train', transform=train_transform)
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
net.load_state_dict(torch.load('saved_model_path'))

print('Starting test...')
scene_number = 1
running_count = 0
running_count_error = 0
running_density_error = 0
for example_index, examples in enumerate(test_dataset_loader):
    if example_index == 120:
        print('Scene {}'.format(scene_number))
        print('Total count: {}'.format(running_count))
        print('Mean count error: {}'.format(running_count_error / 120))
        print('Mean density error: {}'.format(running_density_error / 120))
        running_count = 0
        running_count_error = 0
        running_density_error = 0
        scene_number += 1
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
    running_count_error += count_loss
    running_density_error += density_loss
print('Scene {}'.format(scene_number))
print('Total count: {}'.format(running_count))
print('Mean count error: {}'.format(running_count_error / 119))
print('Mean density error: {}'.format(running_density_error / 119))


print('Finished test.')
