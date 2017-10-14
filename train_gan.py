"""
Main code for a GAN training session.
"""
import datetime
import os
import numpy as np
import torch.utils.data
import torchvision
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler

import settings
import transforms
import viewer
from crowd_dataset import CrowdDataset
from hardware import gpu, cpu
from model import Generator, JointCNN


train_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.NegativeOneToOneNormalizeImage(),
                                                  transforms.NumpyArraysToTorchTensors()])
validation_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
                                                       transforms.NegativeOneToOneNormalizeImage(),
                                                       transforms.NumpyArraysToTorchTensors()])

train_dataset = CrowdDataset(settings.database_path, 'train', transform=train_transform)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True,
                                                   num_workers=settings.number_of_data_loader_workers)
validation_dataset = CrowdDataset(settings.database_path, 'validation', transform=validation_transform)
validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=settings.batch_size,
                                                        shuffle=False,
                                                        num_workers=settings.number_of_data_loader_workers)

generator = Generator()
discriminator = JointCNN()
gpu(generator)
gpu(discriminator)
generator_optimizer = Adam(generator.parameters())
discriminator_optimizer = Adam(discriminator.parameters())

step = 0
epoch = 0

discriminator_optimizer.param_groups[0].update({'lr': settings.initial_learning_rate, 'weight_decay': settings.weight_decay})
discriminator_scheduler = lr_scheduler.LambdaLR(discriminator_optimizer, lr_lambda=settings.learning_rate_multiplier_function)
discriminator_scheduler.step(epoch)
generator_optimizer.param_groups[0].update({'lr': settings.initial_learning_rate, 'weight_decay': settings.weight_decay})
generator_scheduler = lr_scheduler.LambdaLR(generator_optimizer, lr_lambda=settings.learning_rate_multiplier_function)
generator_scheduler.step(epoch)

summary_step_period = settings.summary_step_period
running_scalars = defaultdict(float)
validation_running_scalars = defaultdict(float)
running_example_count = 0
datetime_string = datetime.datetime.now().strftime("y%Ym%md%dh%Hm%Ms%S")
trial_directory = os.path.join(settings.log_directory, settings.trial_name + ' ' + datetime_string)
os.makedirs(trial_directory, exist_ok=True)
summary_writer = SummaryWriter(os.path.join(trial_directory, 'train'))
validation_summary_writer = SummaryWriter(os.path.join(trial_directory, 'validation'))
print('Starting training...')
while epoch < settings.number_of_epochs:
    for examples in train_dataset_loader:
        images, labels, _ = examples
        images, labels = Variable(gpu(images)), Variable(gpu(labels))
        z = torch.randn(images.data.shape[0], 100)
        generated_images = generator(Variable(gpu(z)))
        predicted_labels, predicted_counts = discriminator(images)
        real_feature_layer = discriminator.feature_layer
        generated_predicted_labels, generated_predicted_counts = discriminator(generated_images)
        generated_feature_layer = discriminator.feature_layer
        density_loss = torch.abs(predicted_labels - labels).pow(settings.loss_order).sum(1).sum(1).mean()
        count_loss = torch.abs(predicted_counts - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
        loss = count_loss + (density_loss * 10)
        generated_density_loss = torch.abs(generated_predicted_labels).pow(settings.loss_order).sum(1).sum(1).mean()
        generated_count_loss = torch.abs(generated_predicted_counts).pow(settings.loss_order).mean()
        generated_discriminator_loss = generated_count_loss + (generated_density_loss * 10)
        generator_density_loss = generated_predicted_labels.sum(1).sum(1).mean()
        generator_count_loss = generated_predicted_counts.mean()
        generator_loss = (generated_count_loss + (generated_density_loss * 10)).neg()
        discriminator_optimizer.zero_grad()
        (loss + generated_discriminator_loss).backward(retain_graph=True)
        discriminator_optimizer.step()
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()
        running_scalars['Loss'] += loss.data[0]
        running_scalars['Count Loss'] += count_loss.data[0]
        running_scalars['Density Loss'] += density_loss.data[0]
        running_scalars['Generated Discriminator Loss'] += generated_discriminator_loss.data[0]
        running_scalars['Generator Loss'] += generator_loss.data[0]
        running_example_count += images.size()[0]
        if step % summary_step_period == 0 and step != 0:
            comparison_image = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),
                                                                          cpu(predicted_labels))
            summary_writer.add_image('Comparison', comparison_image, global_step=step)
            generated_images_image = torchvision.utils.make_grid(generated_images.data[:9], nrow=3)
            summary_writer.add_image('Generated', generated_images_image, global_step=step)
            mean_loss = running_scalars['Loss'] / running_example_count
            print('[Epoch: {}, Step: {}] Loss: {:g}'.format(epoch, step, mean_loss))
            for name, running_scalar in running_scalars.items():
                mean_scalar = running_scalar / running_example_count
                summary_writer.add_scalar(name, mean_scalar, global_step=step)
                running_scalars[name] = 0
            for validation_examples in train_dataset_loader:
                images, labels, _ = validation_examples
                images, labels = Variable(gpu(images)), Variable(gpu(labels))
                predicted_labels, predicted_counts = discriminator(images)
                density_loss = torch.abs(predicted_labels - labels).pow(settings.loss_order).sum(1).sum(1).mean()
                count_loss = torch.abs(predicted_counts - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
                validation_running_scalars['Density Loss'] += density_loss.data[0]
                validation_running_scalars['Count Loss'] += count_loss.data[0]
            comparison_image = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),
                                                                          cpu(predicted_labels))
            validation_summary_writer.add_image('Comparison', comparison_image, global_step=step)
            for name, running_scalar in validation_running_scalars.items():
                mean_scalar = running_scalar / len(validation_dataset)
                validation_summary_writer.add_scalar(name, mean_scalar, global_step=step)
                validation_running_scalars[name] = 0
        step += 1
    epoch += 1
    discriminator_scheduler.step(epoch)
    generator_scheduler.step(epoch)
print('Finished Training')
