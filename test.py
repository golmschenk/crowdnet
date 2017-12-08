"""
Code for a test session.
"""
import csv
import os
from collections import namedtuple
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import scipy.misc

import transforms
import settings as settings_
from crowd_dataset import CrowdDataset
from model import JointCNN, load_trainer
from hardware import gpu, cpu


ExamplePatchWithMeta = namedtuple('ExamplePatchWithMeta', ['example', 'half_patch_size', 'x', 'y'])


def batches_of_examples_with_meta(full_example, batch_size=20):
    """
    Generator for example patch batches with meta.

    :param full_example: The original full size example.
    :type full_example: CrowdExample
    :param batch_size: The number of patches per batch to get.
    :type batch_size: int
    :return: A list of examples patches with meta.
    :rtype: list[ExamplePatchWithMeta]
    """
    patch_transform = transforms.ExtractPatchForPositionAndRescale()
    test_transform = torchvision.transforms.Compose([transforms.NegativeOneToOneNormalizeImage(),
                                                     transforms.NumpyArraysToTorchTensors()])
    sample_x = 0
    sample_y = 0
    half_patch_size = 0  # Don't move on the first patch.
    while True:
        batch = []
        for _ in range(batch_size):
            sample_x += half_patch_size
            if sample_x >= full_example.label.shape[1]:
                sample_x = 0
                sample_y += half_patch_size
            if sample_y >= full_example.label.shape[0]:
                if batch:
                    yield batch
                return
            example_patch, original_patch_size = patch_transform(full_example, sample_y, sample_x)
            example = test_transform(example_patch)
            half_patch_size = int(original_patch_size // 2)
            example_with_meta = ExamplePatchWithMeta(example, half_patch_size, sample_x, sample_y)
            batch.append(example_with_meta)
        yield batch


def test(settings=None):
    """Main script for testing a model."""
    if not settings:
        settings = settings_

    test_dataset = CrowdDataset(settings.test_dataset_path, 'test')

    net = JointCNN()
    model_state_dict, _, _, _ = load_trainer(prefix='discriminator')
    net.load_state_dict(model_state_dict)
    gpu(net)
    net.eval()

    count_errors = []
    density_errors = []
    print('Starting test...')
    scene_number = 1
    running_count = 0
    running_count_error = 0
    running_density_error = 0
    for full_example_index, full_example in enumerate(test_dataset):
        print('Processing example {}'.format(full_example_index), end='\r')
        sum_density_label = np.zeros_like(full_example.label, dtype=np.float32)
        sum_count_label = np.zeros_like(full_example.label, dtype=np.float32)
        hit_predicted_label = np.zeros_like(full_example.label, dtype=np.int32)
        for batch in batches_of_examples_with_meta(full_example):
            images = torch.stack([example_with_meta.example.image for example_with_meta in batch])
            rois = torch.stack([example_with_meta.example.roi for example_with_meta in batch])
            images = Variable(gpu(images))
            predicted_labels, predicted_counts = net(images)
            predicted_labels = predicted_labels * Variable(gpu(rois))
            predicted_labels = cpu(predicted_labels.data).numpy()
            predicted_counts = cpu(predicted_counts.data).numpy()
            for example_index, example_with_meta in enumerate(batch):
                predicted_label = predicted_labels[example_index]
                predicted_count = predicted_counts[example_index]
                x, y = example_with_meta.x, example_with_meta.y
                half_patch_size = example_with_meta.half_patch_size
                predicted_label_sum = np.sum(predicted_label)
                original_patch_dimensions = ((2 * half_patch_size) + 1, (2 * half_patch_size) + 1)
                predicted_label = scipy.misc.imresize(predicted_label, original_patch_dimensions, mode='F')
                unnormalized_predicted_label_sum = np.sum(predicted_label)
                if unnormalized_predicted_label_sum != 0:
                    density_label = predicted_label * predicted_label_sum / unnormalized_predicted_label_sum
                    count_label = predicted_label * predicted_count / unnormalized_predicted_label_sum
                else:
                    density_label = predicted_label
                    count_label = np.full(predicted_label.shape, predicted_count / predicted_label.size)
                y_start_offset = 0
                if y - half_patch_size < 0:
                    y_start_offset = half_patch_size - y
                y_end_offset = 0
                if y + half_patch_size >= full_example.label.shape[0]:
                    y_end_offset = y + half_patch_size + 1 - full_example.label.shape[0]
                x_start_offset = 0
                if x - half_patch_size < 0:
                    x_start_offset = half_patch_size - x
                x_end_offset = 0
                if x + half_patch_size >= full_example.label.shape[1]:
                    x_end_offset = x + half_patch_size + 1 - full_example.label.shape[1]
                sum_density_label[y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                  x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                  ] += density_label[y_start_offset:density_label.shape[0] - y_end_offset,
                                                     x_start_offset:density_label.shape[1] - x_end_offset]
                sum_count_label[y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                ] += count_label[y_start_offset:count_label.shape[0] - y_end_offset,
                                                 x_start_offset:count_label.shape[1] - x_end_offset]
                hit_predicted_label[y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                    x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                    ] += 1
        sum_density_label *= full_example.roi
        sum_count_label *= full_example.roi
        full_predicted_label = sum_density_label / hit_predicted_label.astype(np.float32)
        full_predicted_count = np.sum(sum_count_label / hit_predicted_label.astype(np.float32))
        label_in_roi = full_example.label * full_example.roi
        density_loss = np.abs(full_predicted_label - label_in_roi).sum()
        count_loss = np.abs(full_predicted_count - label_in_roi.sum())
        running_count += full_example.label.sum()
        running_count_error += count_loss
        running_density_error += density_loss
        if ((full_example_index + 1) % 120) == 0:
            print('Scene {}'.format(scene_number))
            print('Total count: {}'.format(running_count))
            count_error = running_count_error / 120
            print('Mean count error: {}'.format(count_error))
            density_error = running_density_error / 120
            print('Mean density error: {}'.format(density_error))
            count_errors.append(count_error)
            density_errors.append(density_error)
            running_count = 0
            running_count_error = 0
            running_density_error = 0
            scene_number += 1

    validation_dataset = CrowdDataset(settings.validation_dataset_path, 'validation')

    print('Starting test...')
    running_count = 0
    running_count_error = 0
    running_density_error = 0
    for full_example_index, full_example in enumerate(validation_dataset):
        print('Processing example {}'.format(full_example_index), end='\r')
        sum_density_label = np.zeros_like(full_example.label, dtype=np.float32)
        sum_count_label = np.zeros_like(full_example.label, dtype=np.float32)
        hit_predicted_label = np.zeros_like(full_example.label, dtype=np.int32)
        for batch in batches_of_examples_with_meta(full_example):
            images = torch.stack([example_with_meta.example.image for example_with_meta in batch])
            rois = torch.stack([example_with_meta.example.roi for example_with_meta in batch])
            images = Variable(gpu(images))
            predicted_labels, predicted_counts = net(images)
            predicted_labels = predicted_labels * Variable(gpu(rois))
            predicted_labels = cpu(predicted_labels.data).numpy()
            predicted_counts = cpu(predicted_counts.data).numpy()
            for example_index, example_with_meta in enumerate(batch):
                predicted_label = predicted_labels[example_index]
                predicted_count = predicted_counts[example_index]
                x, y = example_with_meta.x, example_with_meta.y
                half_patch_size = example_with_meta.half_patch_size
                predicted_label_sum = np.sum(predicted_label)
                original_patch_dimensions = ((2 * half_patch_size) + 1, (2 * half_patch_size) + 1)
                predicted_label = scipy.misc.imresize(predicted_label, original_patch_dimensions, mode='F')
                unnormalized_predicted_label_sum = np.sum(predicted_label)
                if unnormalized_predicted_label_sum != 0:
                    density_label = predicted_label * predicted_label_sum / unnormalized_predicted_label_sum
                    count_label = predicted_label * predicted_count / unnormalized_predicted_label_sum
                else:
                    density_label = predicted_label
                    count_label = np.full(predicted_label.shape, predicted_count / predicted_label.size)
                y_start_offset = 0
                if y - half_patch_size < 0:
                    y_start_offset = half_patch_size - y
                y_end_offset = 0
                if y + half_patch_size >= full_example.label.shape[0]:
                    y_end_offset = y + half_patch_size + 1 - full_example.label.shape[0]
                x_start_offset = 0
                if x - half_patch_size < 0:
                    x_start_offset = half_patch_size - x
                x_end_offset = 0
                if x + half_patch_size >= full_example.label.shape[1]:
                    x_end_offset = x + half_patch_size + 1 - full_example.label.shape[1]
                sum_density_label[y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                  x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                  ] += density_label[y_start_offset:density_label.shape[0] - y_end_offset,
                                                     x_start_offset:density_label.shape[1] - x_end_offset]
                sum_count_label[y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                ] += count_label[y_start_offset:count_label.shape[0] - y_end_offset,
                                                 x_start_offset:count_label.shape[1] - x_end_offset]
                hit_predicted_label[y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                    x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                    ] += 1
        sum_density_label *= full_example.roi
        sum_count_label *= full_example.roi
        full_predicted_label = sum_density_label / hit_predicted_label.astype(np.float32)
        full_predicted_count = np.sum(sum_count_label / hit_predicted_label.astype(np.float32))
        label_in_roi = full_example.label * full_example.roi
        density_loss = np.abs(full_predicted_label - label_in_roi).sum()
        count_loss = np.abs(full_predicted_count - label_in_roi.sum())
        running_count += full_example.label.sum()
        running_count_error += count_loss
        running_density_error += density_loss
    validation_count_error = running_count_error / len(validation_dataset)

    csv_file_path = os.path.join(settings.log_directory, 'Test 2 Results.csv')
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Run Name', 'Scene 1', 'Scene 2', 'Scene 3', 'Scene 4', 'Scene 5', 'Mean',
                             'Scene 1 Density', 'Scene 2 Density', 'Scene 3 Density', 'Scene 4 Density',
                             'Scene 5 Density', 'Mean Density', 'Mean Validation'])
    with open(csv_file_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        path_list = os.path.normpath(settings.load_model_path).split(os.sep)
        model_name = os.path.join(*path_list[-2:])
        test_results = [model_name, *count_errors, np.mean(count_errors),
                        *density_errors, np.mean(density_errors), validation_count_error]
        writer.writerow(test_results)

    print('Finished test.')
    settings.load_model_path = None
    return np.mean(count_errors)


if __name__ == '__main__':
    test()
