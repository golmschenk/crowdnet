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

class VisualizationExport:
    def __init__(self, test_set='test'):
        self.test_set = test_set
        self.visualization_directory = os.path.join('visualization', test_set)
        self.settings = Settings()
        self.datasets_file = self.settings.datasets_json

    def predicted_export(self):
        """
        Save the prediction statistics to NumPy files.
        """
        os.makedirs(self.visualization_directory, exist_ok=True)
        net = CrowdNet()
        if self.test_set == 'train':
            self.switch_file()
            net.settings.datasets_json = self.settings.datasets_json
        net.test()

        predicted_labels_save_path = os.path.join(self.visualization_directory, 'predicted_labels')
        print('Saving {}.npy...'.format(predicted_labels_save_path))
        np.save(predicted_labels_save_path, np.squeeze(net.predicted_test_labels))

        true_labels_save_path = os.path.join(self.visualization_directory, 'true_labels')
        print('Saving {}.npy...'.format(true_labels_save_path))
        np.save(true_labels_save_path, np.squeeze(net.true_labels))

        average_loss_save_path = os.path.join(self.visualization_directory, 'average_loss')
        print('Saving {}.npy...'.format(average_loss_save_path))
        np.save(average_loss_save_path, net.predicted_test_labels_average_loss[1:])

        person_count_save_path = os.path.join(self.visualization_directory, 'predicted_person_count')
        print('Saving {}.npy...'.format(person_count_save_path))
        np.save(person_count_save_path, net.predicted_test_labels_person_count[1:])

        relative_miscount_count_save_path = os.path.join(self.visualization_directory, 'relative_miscount_count')
        print('Saving {}.npy...'.format(relative_miscount_count_save_path))
        np.save(relative_miscount_count_save_path, net.predicted_test_labels_relative_miscount[1:])

        net.reset_graph()

    def true_export(self):
        """
        Save the true statistics to NumPy files.
        """
        tfrecords_processor = TFRecordsProcessor()
        with open(self.datasets_file) as datasets_json:
            dataset_file_names = json.load(datasets_json)
        images, labels = tfrecords_processor.read_to_numpy(os.path.join(self.settings.data_directory,
                                                                        dataset_file_names['test'][0]))
        for test_file_name in dataset_file_names['test'][1:]:
            file_images, file_labels = tfrecords_processor.read_to_numpy(os.path.join(self.settings.data_directory,
                                                                                      test_file_name))
            images = np.concatenate([images, file_images], axis=0)
            labels = np.concatenate([labels, file_labels], axis=0)
        person_count = np.sum(labels, axis=(1, 2))

        images_save_path = os.path.join(self.visualization_directory, 'images')
        print('Saving {}.npy...'.format(images_save_path))
        np.save(images_save_path, images)

        person_count_save_path = os.path.join(self.visualization_directory, 'true_person_count')
        print('Saving {}.npy...'.format(person_count_save_path))
        np.save(person_count_save_path, person_count)

    def generate_result_images(self):
        """
        Generates the images for displaying in the visualization.
        """
        os.makedirs(os.path.join(self.visualization_directory, 'predicted_label_images'), exist_ok=True)
        os.makedirs(os.path.join(self.visualization_directory, 'true_label_images'), exist_ok=True)
        os.makedirs(os.path.join(self.visualization_directory, 'original_images'), exist_ok=True)
        images = np.load(os.path.join(self.visualization_directory, 'images.npy'))
        predicted_labels = np.load(os.path.join(self.visualization_directory, 'predicted_labels.npy'))
        true_labels = np.load(os.path.join(self.visualization_directory, 'true_labels.npy'))
        assert images.shape[0] == predicted_labels.shape[0]
        assert images.shape[0] == true_labels.shape[0]
        for index, image in enumerate(images):
            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(self.visualization_directory, 'original_images', '{}.jpeg'.format(index)))
        for index, predicted_label in enumerate(predicted_labels):
            normalized = (predicted_label - np.min(predicted_label)) / (np.max(predicted_label) - np.min(predicted_label))
            cmap = cm.get_cmap('jet')
            pil_image = Image.fromarray(cmap(normalized, bytes=True))
            pil_image.save(os.path.join(self.visualization_directory, 'predicted_label_images', '{}.jpeg'.format(index)))
        for index, true_label in enumerate(true_labels):
            normalized = (true_label - np.min(true_label)) / (np.max(true_label) - np.min(true_label))
            cmap = cm.get_cmap('jet')
            pil_image = Image.fromarray(cmap(normalized, bytes=True))
            pil_image.save(os.path.join(self.visualization_directory, 'true_label_images', '{}.jpeg'.format(index)))

    def switch_file(self):
        with open(self.settings.datasets_json) as original_json_file:
            datasets = json.load(original_json_file)
        datasets['test'], datasets['train'] = datasets['train'], datasets['test']
        self.settings.datasets_json = 'tmp_' + self.settings.datasets_json
        with open(self.settings.datasets_json, 'w') as swap_json_file:
            json.dump(datasets, swap_json_file)

    def export(self):
        self.predicted_export()
        self.true_export()
        self.generate_result_images()

    @classmethod
    def export_test_and_train(cls):
        cls(test_set='test').export()
        cls(test_set='train').export()

if __name__ == '__main__':
    VisualizationExport.export_test_and_train()
