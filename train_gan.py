"""
Main code for a GAN training session.
"""
import datetime
import os
import torch.utils.data
import torchvision
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler, Adam

import settings
import transforms
import viewer
from crowd_dataset import CrowdDataset, CrowdDatasetWithUnlabeled
from hardware import gpu, cpu
from model import Generator, JointCNN, load_trainer, save_trainer, Predictor


def main(trial_name=None, train_dataset_path=None):
    if trial_name:
        settings.trial_name = trial_name
    if train_dataset_path:
        settings.train_dataset_path = train_dataset_path
    train_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.NegativeOneToOneNormalizeImage(),
                                                      transforms.NumpyArraysToTorchTensors()])
    validation_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
                                                           transforms.NegativeOneToOneNormalizeImage(),
                                                           transforms.NumpyArraysToTorchTensors()])

    train_dataset = CrowdDatasetWithUnlabeled(settings.train_dataset_path, 'train', transform=train_transform)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True,
                                                       num_workers=settings.number_of_data_loader_workers)
    validation_dataset = CrowdDataset(settings.validation_dataset_path, 'validation', transform=validation_transform)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=settings.batch_size,
                                                            shuffle=False,
                                                            num_workers=settings.number_of_data_loader_workers)

    generator = Generator()
    discriminator = JointCNN()
    predictor = Predictor()
    gpu(generator)
    gpu(discriminator)
    gpu(predictor)
    generator_optimizer = Adam(generator.parameters())
    discriminator_optimizer = Adam(discriminator.parameters())
    predictor_optimizer = Adam(discriminator.parameters())

    step = 0
    epoch = 0

    if settings.load_model_path:
        d_model_state_dict, d_optimizer_state_dict, epoch, step = load_trainer(prefix='discriminator')
        discriminator.load_state_dict(d_model_state_dict)
        discriminator_optimizer.load_state_dict(d_optimizer_state_dict)
    discriminator_optimizer.param_groups[0].update({'lr': settings.initial_learning_rate,
                                                    'weight_decay': settings.weight_decay})
    discriminator_scheduler = lr_scheduler.LambdaLR(discriminator_optimizer,
                                                    lr_lambda=settings.learning_rate_multiplier_function)
    discriminator_scheduler.step(epoch)
    if settings.load_model_path:
        g_model_state_dict, g_optimizer_state_dict, _, _ = load_trainer(prefix='generator')
        generator.load_state_dict(g_model_state_dict)
        generator_optimizer.load_state_dict(g_optimizer_state_dict)
    generator_optimizer.param_groups[0].update({'lr': settings.initial_learning_rate})
    generator_scheduler = lr_scheduler.LambdaLR(generator_optimizer, lr_lambda=settings.learning_rate_multiplier_function)
    generator_scheduler.step(epoch)
    predictor_optimizer.param_groups[0].update({'lr': settings.initial_learning_rate})
    predictor_scheduler = lr_scheduler.LambdaLR(predictor_optimizer,
                                                lr_lambda=settings.learning_rate_multiplier_function)
    predictor_scheduler.step(epoch)

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
        for examples, unlabeled_examples in train_dataset_loader:
            # Real image discriminator processing.
            discriminator_optimizer.zero_grad()
            images, labels, _ = examples
            images, labels = Variable(gpu(images)), Variable(gpu(labels))
            predicted_labels, predicted_counts = discriminator(images)
            real_feature_layer = discriminator.feature_layer
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
            predictor_predicted_counts = predictor(predicted_counts.detach())
            predictor_count_loss = torch.abs(predictor_predicted_counts - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
            predictor_count_loss.backward(predictor.parameters())
            predictor_optimizer.step()
            running_scalars['Predictor/Count Loss'] += predictor_count_loss.data[0]
            running_scalars['Predictor/Count MAE'] += torch.abs(predictor_predicted_counts - labels.sum(1).sum(1)).mean()
            running_scalars['Predictor/Count ME'] += (predictor_predicted_counts - labels.sum(1).sum(1)).mean().data[0]
            # Unlabeled image discriminator processing.
            unlabeled_images, _, _ = unlabeled_examples
            unlabeled_images = Variable(gpu(unlabeled_images))
            unlabeled_predicted_labels, unlabeled_predicted_counts = discriminator(unlabeled_images)
            label_count_mean = labels.sum(1).sum(1).mean()
            count_mean = labels.sum(1).sum(1).mean()
            unlabeled_predicted_count_mean = unlabeled_predicted_counts.mean()
            unlabeled_predicted_label_count_mean = unlabeled_predicted_labels.sum(1).sum(1).mean()
            beta = 2.0
            # noinspection PyArgumentList
            zero = Variable(gpu(torch.FloatTensor([0])))
            unlabeled_count_loss_min = torch.max(zero, count_mean / beta - unlabeled_predicted_count_mean)
            unlabeled_count_loss_max = torch.max(zero, unlabeled_predicted_count_mean - count_mean * beta)
            unlabeled_label_loss_min = torch.max(zero,
                                                 label_count_mean / beta - unlabeled_predicted_label_count_mean)
            unlabeled_label_loss_max = torch.max(zero,
                                                 unlabeled_predicted_label_count_mean - label_count_mean * beta)
            unlabeled_density_loss = unlabeled_label_loss_max + unlabeled_label_loss_min
            unlabeled_count_loss = unlabeled_count_loss_max + unlabeled_count_loss_min
            unlabeled_loss = unlabeled_count_loss + (unlabeled_density_loss * 10)
            running_scalars['Unlabeled/Count ME'] += (unlabeled_predicted_count_mean - count_mean).data[0]
            running_scalars['Unlabeled/Count'] += unlabeled_predicted_count_mean.data[0]
            running_scalars['Unlabeled/Loss'] += unlabeled_loss.data[0]
            unlabeled_loss.backward()
            # Fake image discriminator processing.
            current_batch_size = images.data.shape[0]
            z = torch.randn(current_batch_size, 100)
            fake_images = generator(Variable(gpu(z)))
            fake_predicted_labels, fake_predicted_counts = discriminator(fake_images)
            fake_density_loss = torch.abs(fake_predicted_labels).pow(settings.loss_order).sum(1).sum(1).mean()
            fake_count_loss = torch.abs(fake_predicted_counts).pow(settings.loss_order).mean()
            fake_mean_count = fake_predicted_counts.mean()
            fake_discriminator_loss = fake_count_loss + (fake_density_loss * 10)
            running_scalars['Fake/Count'] += fake_mean_count.data[0]
            running_scalars['Fake/Loss'] += fake_discriminator_loss.data[0]
            fake_discriminator_loss.backward(retain_graph=True)
            # Gradient penalty.
            alpha = Variable(gpu(torch.rand(3, current_batch_size, 1, 1, 1)))
            alpha = alpha / alpha.sum(0)
            interpolates = alpha[0] * images + alpha[1] * unlabeled_images + alpha[2] * fake_images
            interpolates_labels, interpolates_counts = discriminator(interpolates)
            density_gradients = torch.autograd.grad(outputs=interpolates_labels, inputs=interpolates,
                                                    grad_outputs=gpu(torch.ones(interpolates_labels.size())),
                                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            density_gradients = density_gradients.view(current_batch_size, -1)
            density_gradient_penalty = ((density_gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            count_gradients = torch.autograd.grad(outputs=interpolates_counts, inputs=interpolates,
                                                  grad_outputs=gpu(torch.ones(interpolates_counts.size())),
                                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
            count_gradients = count_gradients.view(current_batch_size, -1)
            count_gradients_penalty = ((count_gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            gradient_penalty = count_gradients_penalty + density_gradient_penalty * 10
            gradient_penalty.backward()
            # Discriminator update.
            discriminator_optimizer.step()
            # Generator image processing.
            generator_optimizer.zero_grad()
            z = torch.randn(current_batch_size, 100)
            fake_images = generator(Variable(gpu(z)))
            _, _ = discriminator(fake_images)  # Produces feature layer for next line.
            fake_feature_layer = discriminator.feature_layer
            detached_predicted_counts = predicted_counts.detach()
            detached_predicted_labels = predicted_labels.detach()
            detached_real_feature_layer = real_feature_layer.detach()
            # noinspection PyArgumentList
            epsilon = Variable(gpu(torch.FloatTensor([1e-10])))
            count_weights = (detached_predicted_counts / torch.max(detached_predicted_counts.sum(), epsilon)).view(-1, 1, 1, 1)
            labels_weights = (detached_predicted_labels.sum(1).sum(1) / torch.max(detached_predicted_labels.sum(), epsilon)).view(-1, 1, 1, 1)
            feature_weights = (count_weights + (labels_weights * 10)) / 11
            weighted_real_feature_layer = feature_weights * detached_real_feature_layer
            generator_loss = (weighted_real_feature_layer.mean(0) - fake_feature_layer.mean(0)).abs().sum()
            running_scalars['Generator/Loss'] += generator_loss.data[0]
            # Generator update.
            if step % 5 == 0:
                generator_loss.backward()
                generator_optimizer.step()

            running_example_count += images.size()[0]
            if step % settings.summary_step_period == 0 and step != 0:
                comparison_image = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),
                                                                              cpu(predicted_labels))
                summary_writer.add_image('Comparison', comparison_image, global_step=step)
                fake_images_image = torchvision.utils.make_grid(fake_images.data[:9], nrow=3)
                summary_writer.add_image('Fake', fake_images_image, global_step=step)
                mean_loss = running_scalars['Labeled/Loss'] / running_example_count
                print('[Epoch: {}, Step: {}] Loss: {:g}'.format(epoch, step, mean_loss))
                for name, running_scalar in running_scalars.items():
                    mean_scalar = running_scalar / running_example_count
                    summary_writer.add_scalar(name, mean_scalar, global_step=step)
                    running_scalars[name] = 0
                running_example_count = 0
                for validation_examples in validation_dataset_loader:
                    images, labels, _ = validation_examples
                    images, labels = Variable(gpu(images)), Variable(gpu(labels))
                    predicted_labels, predicted_counts = discriminator(images)
                    density_loss = torch.abs(predicted_labels - labels).pow(settings.loss_order).sum(1).sum(1).mean()
                    count_loss = torch.abs(predicted_counts - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
                    count_mae = torch.abs(predicted_counts - labels.sum(1).sum(1)).mean()
                    count_me = (predicted_counts - labels.sum(1).sum(1)).mean()
                    validation_running_scalars['Labeled/Density Loss'] += density_loss.data[0]
                    validation_running_scalars['Labeled/Count Loss'] += count_loss.data[0]
                    validation_running_scalars['Labeled/Count MAE'] += count_mae.data[0]
                    validation_running_scalars['Labeled/Count ME'] += count_me.data[0]
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
        predictor_scheduler.step(epoch)
        if epoch != 0 and epoch % settings.save_epoch_period == 0:
            save_trainer(trial_directory, discriminator, discriminator_optimizer, epoch, step, prefix='discriminator')
            save_trainer(trial_directory, generator, generator_optimizer, epoch, step, prefix='generator')
    save_trainer(trial_directory, discriminator, discriminator_optimizer, epoch, step, prefix='discriminator')
    save_trainer(trial_directory, generator, generator_optimizer, epoch, step, prefix='generator')
    print('Finished Training')
    return trial_directory

if __name__ == '__main__':
    main()
