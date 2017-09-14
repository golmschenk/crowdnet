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
from pytorch_crowd_dataset import CrowdDataset

run_name = 'Joint CNN'

train_transform = torchvision.transforms.Compose([transforms.Rescale([564 // 8, 720 // 8]),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.NegativeOneToOneNormalizeImage(),
                                                  transforms.NumpyArraysToTorchTensors()])
validation_transform = torchvision.transforms.Compose([transforms.Rescale([564 // 8, 720 // 8]),
                                                       transforms.NegativeOneToOneNormalizeImage(),
                                                       transforms.NumpyArraysToTorchTensors()])

train_dataset = CrowdDataset('../storage/data/world_expo_datasets', 'train', transform=train_transform)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
validation_dataset = CrowdDataset('../storage/data/world_expo_datasets', 'validation', transform=validation_transform)
validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=False, num_workers=4)


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
criterion = L1Loss()
optimizer = Adam(net.parameters())

summary_step_period = 100

step = 0
running_loss = 0
running_example_count = 0
datetime_string = datetime.datetime.now().strftime("y%Ym%md%dh%Hm%Ms%S")
log_path_name = os.path.join('../storage/logs', run_name + ' {} ' + datetime_string)
summary_writer = SummaryWriter(log_path_name.format('train'))
validation_summary_writer = SummaryWriter(log_path_name.format('validation'))
print('Starting training...')
for epoch in range(summary_step_period):
    for examples in train_dataset_loader:
        images, labels, roi = examples
        images, labels, roi = Variable(images), Variable(labels), Variable(roi)
        predicted_density_maps, predicted_count_maps = net(images)
        predicted_density_maps, predicted_count_maps = predicted_density_maps.squeeze(dim=1), predicted_count_maps.squeeze(dim=1)
        predicted_density_maps = predicted_density_maps * roi
        predicted_count_maps = predicted_count_maps * roi
        density_loss = criterion(predicted_density_maps, labels)
        count_loss = criterion(predicted_count_maps.sum(1).sum(1), labels.sum(1).sum(1))
        loss = density_loss + count_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.data[0]
        running_example_count += images.size()[0]
        if step % summary_step_period == 0 and step != 0:
            comparison_image = viewer.create_crowd_images_comparison_grid(images, labels, predicted_density_maps)
            summary_writer.add_image('Comparison', comparison_image, global_step=step)
            mean_loss = running_loss / running_example_count
            print('[Epoch: {}, Step: {}] Loss: {:g}'.format(epoch, step, mean_loss))
            summary_writer.add_scalar('Loss', mean_loss, global_step=step)
            running_loss = 0
            running_example_count = 0
            validation_density_running_loss = 0
            validation_count_running_loss = 0
            for validation_examples in train_dataset_loader:
                images, labels, roi = validation_examples
                images, labels, roi = Variable(images), Variable(labels), Variable(roi)
                predicted_density_maps, predicted_count_maps = net(images).squeeze(dim=1)
                predicted_density_maps = predicted_density_maps * roi
                predicted_count_maps = predicted_count_maps * roi
                density_loss = criterion(predicted_density_maps, labels)
                count_loss = criterion(predicted_count_maps.sum(1).sum(1), labels.sum(1).sum(1))
                validation_density_running_loss += density_loss.data[0]
                validation_count_running_loss += count_loss.data[0]
            comparison_image = viewer.create_crowd_images_comparison_grid(images, labels, predicted_density_maps)
            validation_summary_writer.add_image('Comparison', comparison_image, global_step=step)
            validation_mean_density_loss = validation_density_running_loss / len(validation_dataset)
            validation_mean_count_loss = validation_count_running_loss / len(validation_dataset)
            validation_summary_writer.add_scalar('Density Loss', validation_mean_density_loss, global_step=step)
            validation_summary_writer.add_scalar('Count Loss', validation_mean_count_loss, global_step=step)
        step += 1

print('Finished Training')
