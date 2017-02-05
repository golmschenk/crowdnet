"""
Code for managing the crowd data.
"""
import numpy as np
import tensorflow as tf
import os

from gonet.data import Data

from settings import Settings


class CrowdData(Data):
    """
    A class for managing the crowd data.
    """

    def __init__(self):
        super().__init__(settings=Settings())

        self.train_size = 'all'
        self.dataset_type = None

    def attain_import_file_paths(self):
        """
        Gets a list of all the file paths for files to be imported.

        :return: The list of the file paths to be imported.
        :rtype: list[str]
        """
        import_file_paths = []
        for file_directory, _, file_names in os.walk(self.settings.import_directory):
            numpy_file_names = [file_name for file_name in file_names if file_name.endswith('.npy')]
            for numpy_file_name in numpy_file_names:
                if 'image' in numpy_file_name:
                    if self.dataset_type == 'deploy':
                        images_path = os.path.abspath(os.path.join(file_directory, numpy_file_name))
                        import_file_paths.append(images_path)
                        continue
                    images_path = os.path.abspath(os.path.join(file_directory, numpy_file_name))
                    if os.path.isfile(images_path.replace('image', 'density')):
                        labels_path = images_path.replace('image', 'density')
                    elif os.path.isfile(images_path.replace('images', 'densities')):
                        labels_path = images_path.replace('images', 'densities')
                    elif os.path.isfile(images_path.replace('image', 'label')):
                        labels_path = images_path.replace('image', 'label')
                    elif os.path.isfile(images_path.replace('images', 'labels')):
                        labels_path = images_path.replace('images', 'labels')
                    else:
                        continue
                    import_file_paths.append((images_path, labels_path))
        return import_file_paths

    def obtain_export_name(self, import_file_path):
        """
        Extracts the name to be used for the export file.

        :param import_file_path: The import path.
        :type import_file_path: str | (str, str)
        :return: The name of the export file.
        :rtype: str
        """
        if self.dataset_type == 'deploy':
            image_file_path = import_file_path
        else:
            image_file_path = import_file_path[0]
        export_name = os.path.splitext(os.path.basename(image_file_path))[0]
        self.data_name = export_name.replace('images', 'data_pairs').replace('image', 'data_pair')

    def import_file(self, file_path):
        """
        Import the data.
        Should be overwritten by subclasses.

        :param file_path: The file path of the file to be imported.
        :type file_path: str | (str, str)
        """
        if self.dataset_type == 'deploy':
            self.import_numpy_images_file(file_path)
        else:
            self.import_numpy_pair_files(file_path)

    def import_numpy_images_file(self, file_path):
        """
        Imports the images from the file path.

        :param file_path: The file path of the numpy image.
        :type file_path: str
        """
        images = np.load(file_path)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        self.images = images
        self.labels = None

    def import_numpy_pair_files(self, file_path_pair):
        """
        Imports the images and labels from the file path pair.

        :param file_path_pair: The pair of file paths.
        :type file_path_pair: (str, str)
        """
        images = np.load(file_path_pair[0])
        labels = np.load(file_path_pair[1]).astype(np.float32)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
            labels = np.expand_dims(labels, axis=0)
        if os.path.isfile(file_path_pair[0].replace('images', 'depth')):
            depth = np.load(file_path_pair[0].replace('images', 'depth'))
            depths = np.tile(depth[np.newaxis, :, :, np.newaxis], (images.shape[0], 1, 1, 1))
            images = np.concatenate((images, depths), axis=3)
        self.images = images
        self.labels = labels

    def preaugmentation_preprocess(self, image, label):
        """
        Preprocesses the image and label to be in the correct format for training.

        :param image: The image to be processed.
        :type image: tf.Tensor
        :param label: The label to be processed.
        :type label: tf.Tensor
        :return: The processed image and label.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        # Choose whether to include depth data (if it's in the TFRecords.
        if image.get_shape()[2] != 3 and self.settings.image_depth == 3:
            image = image[:, :, :3]
        image = tf.image.resize_images(image, [self.settings.image_height, self.settings.image_width])
        resized_label = tf.image.resize_images(label, [self.settings.image_height, self.settings.image_width])
        # Normalize the label to have the same sum as before resizing.
        label_sum = tf.reduce_sum(label)
        resized_label_sum = tf.reduce_sum(resized_label)
        label = tf.cond(
            tf.not_equal(resized_label_sum, 0),
            lambda: (resized_label / resized_label_sum) * label_sum,
            lambda: resized_label
        )
        return image, label


if __name__ == '__main__':
    data = CrowdData()
    data.generate_all_tfrecords()
