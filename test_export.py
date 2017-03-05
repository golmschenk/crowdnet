"""
Code for viewing the data.
"""
import json
import multiprocessing
import os
import numpy as np

from gonet.tfrecords_processor import TFRecordsProcessor

from crowd_net import CrowdNet
from settings import Settings


def predicted_export():
    """
    Save the prediction statistics to NumPy files.
    """
    net = CrowdNet()
    net.test()


def true_export():
    """
    Save the true statistics to NumPy files.
    """
    tfrecords_processor = TFRecordsProcessor()
    settings = Settings()
    with open(settings.datasets_json) as datasets_json:
        dataset_file_names = json.load(datasets_json)
    images, labels = tfrecords_processor.read_to_numpy(os.path.join(settings.data_directory,
                                                                    dataset_file_names['test'][0]))
    for test_file_name in dataset_file_names['test'][1:]:
        file_images, file_labels = tfrecords_processor.read_to_numpy(os.path.join(settings.data_directory,
                                                                                  test_file_name))
        images = np.concatenate([images, file_images], axis=0)
        labels = np.concatenate([labels, file_labels], axis=0)
    person_count = np.sum(labels, axis=(1, 2))

    labels_save_path = 'true_labels'
    print('Saving {}.npy...'.format(labels_save_path))
    np.save(labels_save_path, labels)

    person_count_save_path = 'true_person_count'
    print('Saving {}.npy...'.format(person_count_save_path))
    np.save(person_count_save_path, person_count)


if __name__ == '__main__':
    predicted_export()
    true_export()
