"""
Code for dealing with the database.
"""

import os
import json
import numpy as np


def data_type_block_dataset_from_structured_database(structured_database_directory, data_type_database_directory,
                                                     dataset_json_file_name):
    """
    Converts from the structured database to single file datasets per data type.

    :param structured_database_directory: The path to the structured database.
    :type structured_database_directory: str
    :param data_type_database_directory: The path where the single file per data type database should be placed.
    :type data_type_database_directory: str
    :param dataset_json_file_name: A JSON file containing the specifications of which parts of the structured database
                                   belong to which data type.
    :type dataset_json_file_name: str
    """
    with open(dataset_json_file_name) as json_file:
        dataset_dict = json.load(json_file)
    os.makedirs(data_type_database_directory, exist_ok=True)
    for data_type, cameras in dataset_dict.items():
        images = None
        labels = None
        rois = None
        perspectives = None
        for camera in cameras:
            camera_directory = os.path.join(structured_database_directory, camera)
            camera_images = np.load(os.path.join(camera_directory, 'images.npy'))
            camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'))
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'))
            camera_perspective = np.load(os.path.join(camera_directory, 'perspective.npy'))
            if images is None:
                images = camera_images
                labels = camera_labels
                rois = np.tile(camera_roi, (labels.shape[0], 1, 1))
                perspectives = np.tile(camera_perspective, (labels.shape[0], 1, 1))
            else:
                images = np.concatenate((images, camera_images), axis=0)
                labels = np.concatenate((labels, camera_labels), axis=0)
                rois = np.concatenate((rois, np.tile(camera_roi, (camera_labels.shape[0], 1, 1))), axis=0)
                perspectives = np.concatenate((perspectives, np.tile(camera_perspective,
                                                                     (camera_labels.shape[0], 1, 1))), axis=0)
        dataset_directory = os.path.join(data_type_database_directory, data_type)
        os.makedirs(dataset_directory, exist_ok=True)
        np.save(os.path.join(dataset_directory, 'images.npy'), images)
        np.save(os.path.join(dataset_directory, 'labels.npy'), labels)
        np.save(os.path.join(dataset_directory, 'rois.npy'), rois)
        np.save(os.path.join(dataset_directory, 'perspectives.npy'), perspectives)


data_type_block_dataset_from_structured_database('../storage/data/Head World Expo Database',
                                                  '../storage/data/Head World Expo Datasets',
                                                  '../storage/data/Head World Expo Database/datasets.json')
