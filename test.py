"""
Code for a test session.
"""
import csv
import os
import numpy as np
from torch.autograd import Variable
import torchvision
import scipy.misc

import transforms
import settings
from crowd_dataset import CrowdDataset
from model import JointCNN
from hardware import load

patch_transform = transforms.ExtractPatchForPositionAndRescale()
test_transform = torchvision.transforms.Compose([transforms.NegativeOneToOneNormalizeImage(),
                                                 transforms.NumpyArraysToTorchTensors()])

test_dataset = CrowdDataset(settings.database_path, 'test')

net = JointCNN()
net.load_state_dict(load(settings.load_model_path))

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
    half_patch_size = 1
    y = 0
    while y < full_example.label.shape[0]:
        x = 0
        while x < full_example.label.shape[1]:
            example_patch, original_patch_size = patch_transform(full_example, y, x)
            example = test_transform(example_patch)
            image, label = Variable(example.image.unsqueeze(0)), Variable(example.label)
            predicted_label, predicted_count = net(image)
            predicted_label = predicted_label.data.squeeze(0).numpy()
            predicted_count = predicted_count.data.squeeze(0).numpy()
            predicted_label_sum = np.sum(predicted_label)
            half_patch_size = int(original_patch_size // 2)
            original_patch_dimensions = ((2 * half_patch_size) + 1, (2 * half_patch_size) + 1)
            predicted_label = scipy.misc.imresize(predicted_label, original_patch_dimensions, mode='F')
            unnormalized_predicted_label_sum = np.sum(predicted_label)
            if unnormalized_predicted_label_sum != 0:
                density_label = predicted_label * predicted_label_sum / unnormalized_predicted_label_sum
                count_label = predicted_label * predicted_count / unnormalized_predicted_label_sum
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
                            ] += density_label[y_start_offset:density_label.shape[0] - y_end_offset,
                                               x_start_offset:density_label.shape[1] - x_end_offset]
            hit_predicted_label[y - half_patch_size + y_start_offset:y + half_patch_size + 1 - y_end_offset,
                                x - half_patch_size + x_start_offset:x + half_patch_size + 1 - x_end_offset
                                ] += 1
            x += half_patch_size
        y += half_patch_size
    sum_density_label *= full_example.roi
    sum_count_label *= full_example.roi
    full_predicted_label = sum_density_label / hit_predicted_label.astype(np.float32)
    full_predicted_count = np.sum(sum_count_label / hit_predicted_label.astype(np.float32))
    density_loss = np.abs(full_predicted_label - full_example.label).sum()
    count_loss = np.abs(full_predicted_count - full_example.label.sum())
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

csv_file_path = os.path.join(settings.log_directory, 'Test Results.csv')
if not os.path.isfile(csv_file_path):
    with open(csv_file_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Run Name', 'Scene 1', 'Scene 2', 'Scene 3', 'Scene 4', 'Scene 5', 'Mean',
                         'Scene 1 Density', 'Scene 2 Density', 'Scene 3 Density', 'Scene 4 Density', 'Scene 5 Density',
                         'Mean Density'])
with open(csv_file_path, 'a') as csv_file:
    writer = csv.writer(csv_file)
    path_list = os.path.normpath(settings.load_model_path).split(os.sep)
    model_name = os.path.join(*path_list[-2:])
    test_results = [model_name, *count_errors, np.mean(count_errors),
                    *density_errors, np.mean(density_errors)]
    writer.writerow(test_results)

print('Finished test.')
