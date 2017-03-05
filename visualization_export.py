"""
Code for viewing the data.
"""
import json
from PIL import Image
import os
from matplotlib import cm
import numpy as np

from gonet.tfrecords_processor import TFRecordsProcessor

from crowd_net import CrowdNet
from settings import Settings


def predicted_export():
    """
    Save the prediction statistics to NumPy files.
    """
    os.makedirs('visualization', exist_ok=True)
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

    images_save_path = os.path.join('visualization', 'images')
    print('Saving {}.npy...'.format(images_save_path))
    np.save(images_save_path, images)

    labels_save_path = os.path.join('visualization', 'true_labels')
    print('Saving {}.npy...'.format(labels_save_path))
    np.save(labels_save_path, labels)

    person_count_save_path = os.path.join('visualization', 'true_person_count')
    print('Saving {}.npy...'.format(person_count_save_path))
    np.save(person_count_save_path, person_count)


def generate_result_images():
    """
    Generates the images for displaying in the visualization.
    """
    os.makedirs(os.path.join('visualization', 'predicted_label_images'), exist_ok=True)
    os.makedirs(os.path.join('visualization', 'true_label_images'), exist_ok=True)
    os.makedirs(os.path.join('visualization', 'original_images'), exist_ok=True)
    images = np.load(os.path.join('visualization', 'images.npy'))
    predicted_labels = np.load(os.path.join('visualization', 'predicted_labels.npy'))
    true_labels = np.load(os.path.join('visualization', 'true_labels.npy'))
    assert images.shape[0] == predicted_labels.shape[0]
    assert images.shape[0] == true_labels.shape[0]
    for index, image in enumerate(images):
        pil_image = Image.fromarray(image)
        pil_image.save(os.path.join('visualization', 'original_images', '{}.jpeg'.format(index)))
    for index, predicted_label in enumerate(predicted_labels):
        normalized = (predicted_label - np.min(predicted_label)) / (np.max(predicted_label) - np.min(predicted_label))
        cmap = cm.get_cmap('jet')
        pil_image = Image.fromarray(cmap(normalized, bytes=True))
        pil_image.save(os.path.join('visualization', 'predicted_label_images', '{}.jpeg'.format(index)))
    for index, true_label in enumerate(true_labels):
        normalized = (true_label - np.min(true_label)) / (np.max(true_label) - np.min(true_label))
        cmap = cm.get_cmap('jet')
        pil_image = Image.fromarray(cmap(normalized, bytes=True))
        pil_image.save(os.path.join('visualization', 'true_label_images', '{}.jpeg'.format(index)))


if __name__ == '__main__':
    predicted_export()
    true_export()
    generate_result_images()
