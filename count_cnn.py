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

run_name = 'Count CNN'

train_transform = torchvision.transforms.Compose([transforms.Rescale([564 // 8, 720 // 8]),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.NegativeOneToOneNormalizeImage(),
                                                  transforms.NumpyArraysToTorchTensors()])
validation_transform = torchvision.transforms.Compose([transforms.Rescale([564 // 8, 720 // 8]),
                                                       transforms.NegativeOneToOneNormalizeImage(),
                                                       transforms.NumpyArraysToTorchTensors()])

train_dataset = CrowdDataset('data', 'new_dataset.json', 'train', transform=train_transform)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
validation_dataset = CrowdDataset('data', 'new_dataset.json', 'validation', transform=validation_transform)
validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=4, shuffle=True, num_workers=2)


class CountCNN(Module):
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
        self.conv6 = Conv2d(self.conv5.out_channels, 1, kernel_size=1)

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
        x = leaky_relu(self.conv6(x))
        return x

net = CountCNN()
criterion = L1Loss()
optimizer = Adam(net.parameters())

summary_step_period = 100

step = 0
running_loss = 0
running_example_count = 0
log_path_name = os.path.join('logs', run_name + ' {} ' + datetime.datetime.now().isoformat(sep=' ', timespec='seconds'))
summary_writer = SummaryWriter(log_path_name.format('train'))
validation_summary_writer = SummaryWriter(log_path_name.format('validation'))
print('Starting training...')
for epoch in range(summary_step_period):
    for examples in train_dataset_loader:
        images, labels, roi = examples
        images, labels, roi = Variable(images), Variable(labels), Variable(roi)
        predicted_labels = net(images).squeeze(dim=1)
        predicted_labels = predicted_labels * roi
        loss = criterion(predicted_labels.sum(), labels.sum())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.data[0]
        running_example_count += images.size()[0]
        if step % summary_step_period == 0 and step != 0:
            comparison_image = viewer.create_crowd_images_comparison_grid(images, labels, predicted_labels)
            summary_writer.add_image('Comparison', comparison_image, global_step=step)
            mean_loss = running_loss / running_example_count
            print('[Epoch: {}, Step: {}] Loss: {:g}'.format(epoch, step, mean_loss))
            summary_writer.add_scalar('Loss', mean_loss, global_step=step)
            running_loss = 0
            running_example_count = 0
            validation_running_loss = 0
            for validation_examples in train_dataset_loader:
                images, labels, roi = validation_examples
                images, labels, roi = Variable(images), Variable(labels), Variable(roi)
                predicted_labels = net(images).squeeze(dim=1)
                predicted_labels = predicted_labels * roi
                validation_loss = criterion(predicted_labels, labels)
                validation_running_loss += validation_loss.data[0]
            comparison_image = viewer.create_crowd_images_comparison_grid(images, labels, predicted_labels)
            validation_summary_writer.add_image('Comparison', comparison_image, global_step=step)
            validation_mean_loss = validation_running_loss / len(validation_dataset)
            validation_summary_writer.add_scalar('Loss', validation_mean_loss, global_step=step)
        step += 1

print('Finished Training')
