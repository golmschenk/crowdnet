"""
Code for the model structures.
"""
import os

import pickle
import torch
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d, BatchNorm2d
from torch.nn.functional import relu, tanh

import settings
from hardware import load


class JointCNN(Module):
    """
    A CNN that produces a density map and a count.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=7, padding=3)
        self.max_pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(self.conv1.out_channels, 32, kernel_size=7, padding=3)
        self.conv2_bn = BatchNorm2d(32)
        self.max_pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv2d(self.conv2.out_channels, 64, kernel_size=5, padding=2)
        self.conv3_bn = BatchNorm2d(64)
        self.conv4 = Conv2d(self.conv3.out_channels, 1000, kernel_size=18)
        self.conv5 = Conv2d(self.conv4.out_channels, 400, kernel_size=1)
        self.count_conv = Conv2d(self.conv5.out_channels, 1, kernel_size=1)
        self.density_conv = Conv2d(self.conv5.out_channels, 324, kernel_size=1)
        self.feature_layer = None

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
        x = self.conv2_bn(relu(self.conv2(x)))
        x = self.max_pool2(x)
        x = self.conv3_bn(relu(self.conv3(x)))
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))
        self.feature_layer = x
        x_count = relu(self.count_conv(x)).view(-1)
        x_density = relu(self.density_conv(x)).view(-1, 18, 18)
        return x_density, x_count


class Generator(Module):
    """
    A generator for producing crowd images.
    """
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = ConvTranspose2d(100, 64, kernel_size=18)
        self.conv_transpose1_bn = BatchNorm2d(64)
        self.conv_transpose2 = ConvTranspose2d(self.conv_transpose1.out_channels, 32, kernel_size=4, stride=2,
                                               padding=1)
        self.conv_transpose2_bn = BatchNorm2d(32)
        self.conv_transpose3 = ConvTranspose2d(self.conv_transpose2.out_channels, 3, kernel_size=4, stride=2,
                                               padding=1)

    def forward(self, z):
        """
        The forward pass of the generator.

        :param z: The input images.
        :type z: torch.autograd.Variable
        :return: Generated images.
        :rtype: torch.autograd.Variable
        """
        z = z.view(-1, 100, 1, 1)
        z = self.conv_transpose1_bn(relu(self.conv_transpose1(z)))
        z = self.conv_transpose2_bn(relu(self.conv_transpose2(z)))
        z = tanh(self.conv_transpose3(z))
        return z

    def __call__(self, *args, **kwargs):
        """
        Defined in subclass just to allow for type hinting.

        :return: The predicted labels.
        :rtype: torch.autograd.Variable
        """
        return super().__call__(*args, **kwargs)


class WeightClipper:
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(-0.1, 0.1)


def save_trainer(trial_directory, model, optimizer, epoch, step):
    """
    Saves all the information needed to continue training.

    :param trial_directory: The directory path to save the data to.
    :type trial_directory: str
    :param model: The model to save.
    :type model: torch.nn.Module
    :param optimizer: The optimizer to save.
    :type optimizer: torch.optim.optimizer.Optimizer
    :param epoch: The number of epochs completed.
    :type epoch: int
    :param step: The number of steps completed.
    :type step: int
    """
    torch.save(model.state_dict(), os.path.join(trial_directory, 'model {}'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(trial_directory, 'optimizer {}'.format(epoch)))
    with open(os.path.join(trial_directory, 'meta {}'.format(epoch)), 'wb') as pickle_file:
        pickle.dump({'epoch': epoch, 'step': step}, pickle_file)


def load_trainer():
    """
    Saves all the information needed to continue training.

    :return: The model and optimizer state dict and the metadata for the training run.
    :rtype: dict[torch.Tensor], dict[torch.Tensor], int, int
    """
    model_state_dict = load(settings.load_model_path)
    optimizer_state_dict = torch.load(settings.load_model_path.replace('model', 'optimizer'))
    with open(settings.load_model_path.replace('model', 'meta'), 'rb') as pickle_file:
        metadata = pickle.load(pickle_file)
    if settings.restore_mode == 'continue':
        step = metadata['step']
        epoch = metadata['epoch']
    else:
        step = 0
        epoch = 0
    return model_state_dict, optimizer_state_dict, epoch, step
