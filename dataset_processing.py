"""
Code for manipulating the dataset.
"""
import os
import random

import numpy as np
from tfrecords_processor import TFRecordsProcessor
import json
import copy

def minify_dataset(data_directory, dataset_json_filename, dataset_name, number_of_cameras=None, images_per_camera=1):
    with open(dataset_json_filename) as dataset_json_file:
        mini_dataset_images = []
        mini_dataset_labels = []
        dataset_json = json.load(dataset_json_file)
        new_dataset_json = copy.deepcopy(dataset_json)
        filenames = dataset_json['train']
        if number_of_cameras:
            random.shuffle(filenames)
            filenames = filenames[:number_of_cameras]
        tfrecords_processor = TFRecordsProcessor()
        new_file_names = []
        for filename in filenames:
            images_numpy, labels_numpy = tfrecords_processor.read_to_numpy(os.path.join(data_directory, filename))
            random_indexes = np.random.choice(labels_numpy.shape[0], images_per_camera, replace=False)
            for index in random_indexes:
                mini_dataset_images.append(images_numpy[index])
                mini_dataset_labels.append(labels_numpy[index])
            label_guess = np.mean(np.sum(np.maximum(labels_numpy[random_indexes], 0), axis=(1, 2)))
            new_unlabled_tfrecords_file_name = dataset_name + '_' + filename
            tfrecords_processor.write_from_numpy(os.path.join(data_directory, new_unlabled_tfrecords_file_name),
                                                 image_shape=images_numpy.shape[1:],
                                                 images=images_numpy,
                                                 label_shape=labels_numpy.shape[1:],
                                                 labels=labels_numpy,
                                                 label_guess=label_guess)
            new_file_names.append(new_unlabled_tfrecords_file_name)
        mini_dataset_images_numpy = np.stack(mini_dataset_images)
        mini_dataset_labels_numpy = np.stack(mini_dataset_labels)
        new_train_tfrecords_file_name = dataset_name + '.tfrecords'
        tfrecords_processor.write_from_numpy(os.path.join(data_directory, new_train_tfrecords_file_name),
                                             image_shape=mini_dataset_images_numpy.shape[1:],
                                             images=mini_dataset_images_numpy,
                                             label_shape=mini_dataset_labels_numpy.shape[1:],
                                             labels=mini_dataset_labels_numpy)
        new_dataset_json['train'] = [new_train_tfrecords_file_name]
        new_dataset_json['unlabeled'] = new_file_names
        with open(os.path.join(data_directory, dataset_name + '.json'), 'w') as new_dataset_json_file:
            json.dump(new_dataset_json, new_dataset_json_file)

minify_dataset('../storage/data', 'world_expo_datasets.json', '1_camera', number_of_cameras=1)
