"""
Main code for a GAN training session.
"""
import datetime
import os
import re

import torch.utils.data
import torchvision
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam

from settings import Settings
import transforms
import viewer
from crowd_dataset import CrowdDataset, CrowdDatasetWithUnlabeled
from hardware import gpu, cpu
from model import GAN, load_trainer, save_trainer


def train(settings=None):
    """Main script for training the semi-supervised GAN."""
    if not settings:
        settings = Settings()
    train_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.NegativeOneToOneNormalizeImage(),
                                                      transforms.NumpyArraysToTorchTensors()])
    validation_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
                                                           transforms.NegativeOneToOneNormalizeImage(),
                                                           transforms.NumpyArraysToTorchTensors()])

    train_dataset = CrowdDataset(settings.train_dataset_path, 'train', transform=train_transform)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True,
                                                       num_workers=settings.number_of_data_loader_workers)
    validation_dataset = CrowdDataset(settings.validation_dataset_path, 'validation', transform=validation_transform)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=settings.batch_size,
                                                            shuffle=False,
                                                            num_workers=settings.number_of_data_loader_workers)

    gan = GAN()
    gpu(gan)
    D = gan.D
    G = gan.G
    P = gan.P
    gpu(P)
    discriminator_optimizer = Adam(D.parameters())
    generator_optimizer = Adam(G.parameters())
    predictor_optimizer = Adam(P.parameters())

    step = 0
    epoch = 0

    if settings.load_model_path:
        d_model_state_dict, d_optimizer_state_dict, epoch, step = load_trainer(prefix='discriminator')
        D.load_state_dict(d_model_state_dict)
        discriminator_optimizer.load_state_dict(d_optimizer_state_dict)
    discriminator_optimizer.param_groups[0].update({'lr': 1e-4, 'weight_decay': settings.weight_decay})
    if settings.load_model_path:
        g_model_state_dict, g_optimizer_state_dict, _, _ = load_trainer(prefix='generator')
        G.load_state_dict(g_model_state_dict)
        generator_optimizer.load_state_dict(g_optimizer_state_dict)
    generator_optimizer.param_groups[0].update({'lr': 1e-4})

    running_scalars = defaultdict(float)
    validation_running_scalars = defaultdict(float)
    running_example_count = 0
    datetime_string = datetime.datetime.now().strftime("y%Ym%md%dh%Hm%Ms%S")
    trial_directory = os.path.join(settings.log_directory, settings.trial_name + ' ' + datetime_string)
    os.makedirs(trial_directory, exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(trial_directory, 'train'))
    validation_summary_writer = SummaryWriter(os.path.join(trial_directory, 'validation'))
    print('Starting training...')
    step_time_start = datetime.datetime.now()
    while epoch < settings.number_of_epochs:
        for examples in train_dataset_loader:
            # Real image discriminator processing.
            discriminator_optimizer.zero_grad()
            images, labels, _ = examples
            images, labels = Variable(gpu(images)), Variable(gpu(labels))
            predicted_labels, predicted_counts = D(images)
            real_feature_layer = D.feature_layer
            density_loss = torch.abs(predicted_labels - labels).pow(settings.loss_order).sum(1).sum(1).mean()
            count_loss = torch.abs(predicted_counts - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
            loss = count_loss + (density_loss * 10)
            loss.backward()
            running_scalars['Labeled/Loss'] += loss.data[0]
            running_scalars['Labeled/Count Loss'] += count_loss.data[0]
            running_scalars['Labeled/Density Loss'] += density_loss.data[0]
            running_scalars['Labeled/Count ME'] += (predicted_counts - labels.sum(1).sum(1)).mean().data[0]
            # Predictor.
            predictor_optimizer.zero_grad()
            predictor_predicted_counts = P(predicted_counts.detach())
            predictor_count_loss = torch.abs(predictor_predicted_counts - labels.sum(1).sum(1)
                                             ).pow(settings.loss_order).mean()
            predictor_count_loss.backward()
            predictor_optimizer.step()
            running_scalars['Predictor/Count Loss'] += predictor_count_loss.data[0]
            running_scalars['Predictor/Count MAE'] += torch.abs(predictor_predicted_counts - labels.sum(1).sum(1)
                                                                ).mean().data[0]
            running_scalars['Predictor/Count ME'] += (predictor_predicted_counts - labels.sum(1).sum(1)).mean().data[0]
            running_scalars['Predictor/Exponent'] += P.exponent.data[0]
            # Discriminator update.
            discriminator_optimizer.step()

            running_example_count += images.size()[0]
            if step % settings.summary_step_period == 0 and step != 0:
                comparison_image = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),
                                                                              cpu(predicted_labels))
                summary_writer.add_image('Comparison', comparison_image, global_step=step)
                print('\rStep {}, {}...'.format(step, datetime.datetime.now() - step_time_start), end='')
                step_time_start = datetime.datetime.now()
                for name, running_scalar in running_scalars.items():
                    mean_scalar = running_scalar / running_example_count
                    summary_writer.add_scalar(name, mean_scalar, global_step=step)
                    running_scalars[name] = 0
                running_example_count = 0
                for validation_examples in validation_dataset_loader:
                    images, labels, _ = validation_examples
                    images, labels = Variable(gpu(images)), Variable(gpu(labels))
                    predicted_labels, predicted_counts = D(images)
                    density_loss = torch.abs(predicted_labels - labels).pow(settings.loss_order).sum(1).sum(1).mean()
                    count_loss = torch.abs(predicted_counts - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
                    count_mae = torch.abs(predicted_counts - labels.sum(1).sum(1)).mean()
                    count_me = (predicted_counts - labels.sum(1).sum(1)).mean()
                    validation_running_scalars['Labeled/Density Loss'] += density_loss.data[0]
                    validation_running_scalars['Labeled/Count Loss'] += count_loss.data[0]
                    validation_running_scalars['Labeled/Count MAE'] += count_mae.data[0]
                    validation_running_scalars['Labeled/Count ME'] += count_me.data[0]
                    predictor_predicted_counts = P(predicted_counts.detach())
                    validation_running_scalars['Predictor/Count MAE'] += torch.abs(predictor_predicted_counts -
                                                                                   labels.sum(1).sum(1)).mean().data[0]
                    validation_running_scalars['Predictor/Count ME'] += (predictor_predicted_counts -
                                                                         labels.sum(1).sum(1)).mean().data[0]
                comparison_image = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),
                                                                              cpu(predicted_labels))
                validation_summary_writer.add_image('Comparison', comparison_image, global_step=step)
                for name, running_scalar in validation_running_scalars.items():
                    mean_scalar = running_scalar / len(validation_dataset)
                    validation_summary_writer.add_scalar(name, mean_scalar, global_step=step)
                    validation_running_scalars[name] = 0
            step += 1
        epoch += 1
        if epoch != 0 and epoch % settings.save_epoch_period == 0:
            save_trainer(trial_directory, D, discriminator_optimizer, epoch, step, prefix='discriminator')
            save_trainer(trial_directory, G, generator_optimizer, epoch, step, prefix='generator')
    save_trainer(trial_directory, D, discriminator_optimizer, epoch, step, prefix='discriminator')
    save_trainer(trial_directory, G, generator_optimizer, epoch, step, prefix='generator')
    print('Finished Training')
    return trial_directory


def clean_scientific_notation(string):
    regex = r'\.?0*e([+\-])0*([0-9])'
    string = re.sub(regex, r'e\g<1>\g<2>', string)
    string = re.sub(r'e\+', r'e', string)
    return string


if __name__ == '__main__':
    for camera_count in [5]:
        for image_count in [5]:
            settings = Settings()
            settings.learning_rate = 1e-5
            settings.trial_name = 'CNN {} Cameras {} Images lr {:e}'.format(camera_count, image_count, settings.learning_rate)
            settings.trial_name = clean_scientific_notation(settings.trial_name)
            print('Processing {}...'.format(settings.trial_name))
            settings.train_dataset_path = '/media/root/Gold/crowd/data/World Expo Datasets/{} Camera {} Images Target Unlabeled'.format(
                camera_count, image_count)
            train(settings)
