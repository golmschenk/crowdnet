"""
Main code for a training session.
"""
import os
import datetime
from torch.autograd import Variable
from torch.nn import Module, Conv2d, L1Loss, MaxPool2d
from torch.nn.functional import leaky_relu, relu
from torch.optim import Adam
import torch.utils.data
import torchvision
from tensorboard import SummaryWriter

import transforms
import viewer
from crowd_dataset import CrowdDataset

run_name = 'Joint CNN'

train_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.NegativeOneToOneNormalizeImage(),
                                                  transforms.NumpyArraysToTorchTensors()])
validation_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
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

    def forward(self, x):
        """
        The forward pass of the network.

        :param x: The input images.
        :type x: torch.autograd.Variable
        :return: The predicted density labels.
        :rtype: torch.autograd.Variable
        """
        x = relu(self.conv1(x))
        x = self.max_pool1(x)
        x = relu(self.conv2(x))
        x = self.max_pool2(x)
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))
        x_count = self.count_conv(x).view(-1)
        x_density = self.density_conv(x).view(-1, 18, 18)
        return x_density, x_count


net = JointCNN()
net.cuda()
optimizer = Adam(net.parameters())

summary_step_period = 100

step = 0
running_loss = 0
count_running_loss = 0
density_running_loss = 0
running_example_count = 0
datetime_string = datetime.datetime.now().strftime("y%Ym%md%dh%Hm%Ms%S")
log_path_name = os.path.join('../storage/logs', run_name + ' {} ' + datetime_string)
summary_writer = SummaryWriter(log_path_name.format('train'))
validation_summary_writer = SummaryWriter(log_path_name.format('validation'))
print('Starting training...')
for epoch in range(100):
    for examples in train_dataset_loader:
        images, labels, _ = examples
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        predicted_density_maps, predicted_count_maps = net(images)
        density_loss = torch.abs(predicted_density_maps - labels).sum(1).sum(1).mean()
        count_loss = torch.abs(predicted_count_maps - labels.sum(1).sum(1)).mean()
        loss = count_loss + (density_loss * 10)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.data[0]
        count_running_loss += count_loss.data[0]
        density_running_loss += density_loss.data[0]
        running_example_count += images.size()[0]
        if step % summary_step_period == 0 and step != 0:
            comparison_image = viewer.create_crowd_images_comparison_grid(images.cpu(), labels.cpu(), predicted_density_maps.cpu())
            summary_writer.add_image('Comparison', comparison_image, global_step=step)
            mean_loss = running_loss / running_example_count
            mean_count_loss = count_running_loss / running_example_count
            mean_density_loss = density_running_loss / running_example_count
            print('[Epoch: {}, Step: {}] Loss: {:g}'.format(epoch, step, mean_loss))
            summary_writer.add_scalar('Loss', mean_loss, global_step=step)
            summary_writer.add_scalar('Count Loss', mean_count_loss, global_step=step)
            summary_writer.add_scalar('Density Loss', mean_density_loss, global_step=step)
            running_loss = 0
            count_running_loss = 0
            density_running_loss = 0
            running_example_count = 0
            validation_density_running_loss = 0
            validation_count_running_loss = 0
            for validation_examples in train_dataset_loader:
                images, labels, _ = validation_examples
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
                predicted_density_maps, predicted_count_maps = net(images)
                density_loss = torch.abs(predicted_density_maps - labels).sum(1).sum(1).mean()
                count_loss = torch.abs(predicted_count_maps - labels.sum(1).sum(1)).mean()
                validation_density_running_loss += density_loss.data[0]
                validation_count_running_loss += count_loss.data[0]
            comparison_image = viewer.create_crowd_images_comparison_grid(images.cpu(), labels.cpu(), predicted_density_maps.cpu())
            validation_summary_writer.add_image('Comparison', comparison_image, global_step=step)
            validation_mean_density_loss = validation_density_running_loss / len(validation_dataset)
            validation_mean_count_loss = validation_count_running_loss / len(validation_dataset)
            validation_summary_writer.add_scalar('Density Loss', validation_mean_density_loss, global_step=step)
            validation_summary_writer.add_scalar('Count Loss', validation_mean_count_loss, global_step=step)
        step += 1

torch.save(net.state_dict(), log_path_name.format('model'))

print('Finished Training')
