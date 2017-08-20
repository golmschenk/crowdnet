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

run_name = 'Basic CNN'

train_transform = torchvision.transforms.Compose([transforms.Rescale([564 // 8, 720 // 8]),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.NormalizeImage(),
                                                  transforms.NumpyArrayToTorchTensor()])
validation_transform = torchvision.transforms.Compose([transforms.Rescale([564 // 8, 720 // 8]),
                                                       transforms.NormalizeImage(),
                                                       transforms.NumpyArrayToTorchTensor()])

train_dataset = CrowdDataset('data', 'new_dataset.json', 'train', transform=train_transform)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
validation_dataset = CrowdDataset('data', 'new_dataset.json', 'validation', transform=validation_transform)
validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=4, shuffle=True, num_workers=2)


class DensityCNN(Module):
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
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = leaky_relu(self.conv3(x))
        x = leaky_relu(self.conv4(x))
        x = leaky_relu(self.conv5(x))
        x = leaky_relu(self.conv6(x))
        return x

net = DensityCNN()
criterion = L1Loss()
optimizer = Adam(net.parameters())

step = 0
log_path_name = os.path.join('logs', run_name + ' ' + datetime.datetime.now().isoformat(sep=' ', timespec='seconds'))
summary_writer = SummaryWriter(log_path_name)
print('Starting training...')
for epoch in range(10):
    running_loss = 0.0
    for examples in train_dataset_loader:
        images, labels, _ = examples
        images, labels = Variable(images), Variable(labels)
        predicted_labels = net(images).squeeze(dim=1)
        loss = criterion(predicted_labels, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.data[0]
        if step % 10 == 0 and step != 0:
            summary_writer.add_image('Comparison', viewer.create_crowd_images_comparison_grid(images, labels, predicted_labels), global_step=step)
            print('[Epoch: {}, Step: {}] Loss: {:g}'.format(epoch, step, running_loss / 100))
            summary_writer.add_scalar('Loss', running_loss / 100, global_step=step)
            running_loss = 0.0
        step += 1

print('Finished Training')
