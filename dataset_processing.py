"""
Code for manipulating the dataset.
"""
import os
import numpy as np
from gonet.tfrecords_processor import TFRecordsProcessor
import json

def minify_dataset(data_directory, dataset_json_filename):
    with open(dataset_json_filename) as dataset_json_file:
        mini_dataset_images = []
        mini_dataset_labels = []
        dataset_json = json.load(dataset_json_file)
        filenames = dataset_json['train']
        tfrecords_processor = TFRecordsProcessor()
        for filename in filenames:
            images_numpy, labels_numpy = tfrecords_processor.read_to_numpy(os.path.join(data_directory, filename))
            middle_index = labels_numpy.shape[0] // 2
            mini_dataset_images.append(images_numpy[middle_index])
            mini_dataset_labels.append(labels_numpy[middle_index])
        mini_dataset_images_numpy = np.stack(mini_dataset_images)
        mini_dataset_labels_numpy = np.stack(mini_dataset_labels)
        tfrecords_processor.write_from_numpy(os.path.join(data_directory, 'mini_dataset'),
                                             image_shape=mini_dataset_images_numpy.shape[1:],
                                             images=mini_dataset_images_numpy,
                                             label_shape=mini_dataset_labels_numpy.shape[1:],
                                             labels=mini_dataset_labels_numpy)
