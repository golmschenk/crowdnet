"""
Main code for a training session.
"""
import datetime
import os
import torch.utils.data
import torchvision
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler

import settings
import transforms
import viewer
from crowd_dataset import CrowdDataset
from hardware import gpu, cpu
from model import JointCNN, load_trainer, save_trainer

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

net = JointCNN()
gpu(net)
optimizer = Adam(net.parameters())
step = 0
epoch = 0

if settings.load_model_path:
    model_state_dict, optimizer_state_dict, epoch, step = load_trainer()
    net.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
optimizer.param_groups[0].update({'lr': settings.initial_learning_rate, 'weight_decay': settings.weight_decay})
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=settings.learning_rate_multiplier_function)
scheduler.step(epoch)

summary_step_period = settings.summary_step_period
running_loss = 0
count_running_loss = 0
density_running_loss = 0
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
        predicted_labels, predicted_counts = net(images)
        density_loss = torch.abs(predicted_labels - labels).pow(settings.loss_order).sum(1).sum(1).mean()
        count_loss = torch.abs(predicted_counts - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
        loss = count_loss + (density_loss * 10)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.data[0]
        count_running_loss += count_loss.data[0]
        density_running_loss += density_loss.data[0]
        running_example_count += images.size()[0]
        if step % summary_step_period == 0 and step != 0:
            comparison_image = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),
                                                                          cpu(predicted_labels))
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
                images, labels = Variable(gpu(images)), Variable(gpu(labels))
                predicted_labels, predicted_counts = net(images)
                density_loss = torch.abs(predicted_labels - labels).pow(settings.loss_order).sum(1).sum(1).mean()
                count_loss = torch.abs(predicted_counts - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
                validation_density_running_loss += density_loss.data[0]
                validation_count_running_loss += count_loss.data[0]
            comparison_image = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),
                                                                          cpu(predicted_labels))
            validation_summary_writer.add_image('Comparison', comparison_image, global_step=step)
            validation_mean_density_loss = validation_density_running_loss / len(validation_dataset)
            validation_mean_count_loss = validation_count_running_loss / len(validation_dataset)
            validation_summary_writer.add_scalar('Density Loss', validation_mean_density_loss, global_step=step)
            validation_summary_writer.add_scalar('Count Loss', validation_mean_count_loss, global_step=step)
        step += 1
    epoch += 1
    scheduler.step(epoch)
    if epoch != 0 and epoch % settings.save_epoch_period == 0:
        save_trainer(trial_directory, net, optimizer, epoch, step)
save_trainer(trial_directory, net, optimizer, epoch, step)
print('Finished Training')
