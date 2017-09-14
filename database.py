"""
Code for dealing with the database.
"""

import os
import json
import numpy as np


def data_type_block_database_from_structured_database(structured_database_directory, data_type_database_directory,
                                                      dataset_json_file_name):
    with open(dataset_json_file_name) as json_file:
        dataset_dict = json.load(json_file)
    os.makedirs(data_type_database_directory, exist_ok=True)
    for data_type, cameras in dataset_dict.items():
        images = None
        labels = None
        rois = None
        for camera in cameras:
            camera_directory = os.path.join(structured_database_directory, camera)
            camera_images = np.load(os.path.join(camera_directory, 'images.npy'))
            camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'))
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'))
            if images is None:
                images = camera_images
                labels = camera_labels
                rois = np.tile(camera_roi, (labels.shape[0], 1, 1))
            else:
                images = np.concatenate((images, camera_images), axis=0)
                labels = np.concatenate((labels, camera_labels), axis=0)
                rois = np.concatenate((np.tile(camera_roi, (camera_labels.shape[0], 1, 1)), rois), axis=0)
        dataset_directory = os.path.join(data_type_database_directory, data_type)
        os.makedirs(dataset_directory, exist_ok=True)
        np.save(os.path.join(dataset_directory, 'images.npy'), images)
        np.save(os.path.join(dataset_directory, 'labels.npy'), labels)
        np.save(os.path.join(dataset_directory, 'rois.npy'), rois)

data_type_block_database_from_structured_database('../storage/data/world_expo_database',
                                                  '../storage/data/world_expo_datasets',
                                                  '../storage/data/world_expo_database/datasets.json')