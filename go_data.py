"""
Code for managing the TFRecord data.
"""
import glob
import os
import h5py
import numpy as np
import tensorflow as tf

from convenience import random_boolean_tensor
from go_tfrecords_reader import GoTFRecordsReader


class GoData:
    """
    A class for managing the TFRecord data.
    """

    def __init__(self):
        self.data_directory = 'data'
        self.data_name = 'nyud_micro'
        self.import_directory = 'data/import'

        # Note, these are *training* sizes. Data will be resized to this size.
        self.image_height = 464 // 8
        self.image_width = 624 // 8
        self.image_depth = 3

        self.images = None
        self.labels = None

        # Internal attributes.
        self._label_height = None
        self._label_width = None
        self._label_depth = None

        os.nice(10)

    @property
    def label_height(self):
        """
        The height of the label data. Defaults to the height of the image.

        :return: Label height.
        :rtype: int
        """
        if self._label_height is None:
            return self.image_height
        return self._label_height

    @label_height.setter
    def label_height(self, value):
        self._label_height = value

    @property
    def label_width(self):
        """
        The width of the label data. Defaults to the width of the image.

        :return: Label width.
        :rtype: int
        """
        if self._label_width is None:
            return self.image_width
        return self._label_width

    @label_width.setter
    def label_width(self, value):
        self._label_width = value

    @property
    def label_depth(self):
        """
        The depth of the label data. Defaults to 1.

        :return: Label depth.
        :rtype: int
        """
        if self._label_depth is None:
            return 1
        return self._label_depth

    @label_depth.setter
    def label_depth(self, value):
        self._label_depth = value

    @property
    def image_shape(self):
        """
        The tuple shape of the image.

        :return: Image shape.
        :rtype: (int, int, int)
        """
        return self.image_height, self.image_width, self.image_depth

    @image_shape.setter
    def image_shape(self, shape):
        self.image_height, self.image_width, self.image_depth = shape

    @property
    def label_shape(self):
        """
        The tuple shape of the label.

        :return: Label shape.
        :rtype: (int, int, int)
        """
        return self.label_height, self.label_width, self.label_depth

    @label_shape.setter
    def label_shape(self, shape):
        self.label_height, self.label_width, self.label_depth = shape

    @property
    def data_path(self):
        """
        Gives the path to the data file.

        :return: The path to the data file.
        :rtype: str
        """
        return os.path.join(self.data_directory, self.data_name)

    @staticmethod
    def read_and_decode_single_example_from_tfrecords(file_name_queue, data_type=None):
        """
        A definition of how TF should read a single example proto from the file record.

        :param file_name_queue: The file name queue to be read.
        :type file_name_queue: tf.QueueBase
        :param data_type: The dataset type being used in.
        :type data_type: str
        :return: The read file data including the image data and label data.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        go_tfrecords_reader = GoTFRecordsReader(file_name_queue, data_type=data_type)
        image = tf.cast(go_tfrecords_reader.image, tf.float32)
        label = go_tfrecords_reader.label

        return image, label

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
        image = tf.image.resize_images(image, self.image_height, self.image_width)
        label = tf.image.resize_images(label, self.image_height, self.image_width)
        return image, label

    @staticmethod
    def postaugmentation_preprocess(image, label):
        """
        Preprocesses the image and label to be in the correct format for training.

        :param image: The image to be processed.
        :type image: tf.Tensor
        :param label: The label to be processed.
        :type label: tf.Tensor
        :return: The processed image and label.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        image = tf.image.per_image_whitening(image)
        return image, label

    @staticmethod
    def horizontally_flip_label(label):
        """
        Changes the label in such a way that it matches its corresponding image if it's been horizontally flipped.
        Should be overridden depending on the type of label data.

        :param label: The label to be "flipped".
        :type label: tf.Tensor
        :return: The "flipped" label.
        :rtype: tf.Tensor
        """
        return tf.image.flip_left_right(label)

    def randomly_flip_horizontally(self, image, label):
        """
        Simultaneously and randomly flips the image and label horizontally, such that they still match after flipping.

        :param image: The image to be flipped (maybe).
        :type image: tf.Tensor
        :param label: The label to be flipped (maybe).
        :type label: tf.Tensor
        :return: The image and label which may be flipped.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        should_flip = random_boolean_tensor()
        image = tf.cond(
            should_flip,
            lambda: tf.image.flip_left_right(image),
            lambda: image
        )
        label = tf.cond(
            should_flip,
            lambda: self.horizontally_flip_label(label),
            lambda: label
        )
        return image, label

    def augment(self, image, label):
        """
        Augments the data in various ways.

        :param image: The image to be augmented.
        :type image: tf.Tensor
        :param label: The label to be augmented
        :type label: tf.Tensor
        :return: The augmented image and label
        :rtype: (tf.Tensor, tf.Tensor)
        """
        # Add Gaussian noise.
        image = image + tf.random_normal(image.get_shape(), mean=0, stddev=8)

        image, label = self.randomly_flip_horizontally(image, label)

        return image, label

    def create_input_tensors_for_dataset(self, data_type, batch_size):
        """
        Prepares the data inputs.

        :param data_type: The type of data file (usually train, validation, or test).
        :type data_type: str
        :param batch_size: The size of the batches
        :type batch_size: int
        :return: The images and depths inputs.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        file_name_queue = self.file_name_queue_for_dataset_directory(data_type)
        image, label = self.read_and_decode_single_example_from_tfrecords(file_name_queue, data_type=data_type)
        image, label = self.preaugmentation_preprocess(image, label)
        if data_type == 'train':
            image, label = self.augment(image, label)
        image, label = self.postaugmentation_preprocess(image, label)

        if data_type in ['test', 'deploy']:
            images, labels = tf.train.batch(
                [image, label], batch_size=batch_size, num_threads=1, capacity=1000 + 3 * batch_size
            )
        else:
            images, labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size, min_after_dequeue=1000
            )

        return images, labels

    def file_name_queue_for_dataset_directory(self, data_type):
        """
        Creates the files name queue for a single TFRecords file.

        :param data_type: The type of dataset being created.
        :type data_type: str
        :return: The file name queue.
        :rtype: tf.QueueBase
        """
        if data_type in ['test', 'deploy']:
            num_epochs = 1
            shuffle = False
        else:
            num_epochs = None
            shuffle = True
        file_paths = []
        for file_path in glob.glob(os.path.join(self.data_directory, data_type, '*.tfrecords')):
            file_paths.append(file_path)
        file_name_queue = tf.train.string_input_producer(file_paths, num_epochs=num_epochs, shuffle=shuffle)
        return file_name_queue

    def convert_mat_file_to_numpy_file(self, mat_file_path, number_of_samples=None):
        """
        Generate image and depth numpy files from the passed mat file path.

        :param mat_file_path: The path to the mat file.
        :type mat_file_path: str
        :param number_of_samples: The number of samples to extract.
        :type number_of_samples: int
        """
        mat_data = h5py.File(mat_file_path, 'r')
        images = self.convert_mat_data_to_numpy_array(mat_data, 'images', number_of_samples=number_of_samples)
        images = self.crop_data(images)
        depths = self.convert_mat_data_to_numpy_array(mat_data, 'depths', number_of_samples=number_of_samples)
        depths = self.crop_data(depths)
        basename = os.path.basename(os.path.splitext(mat_file_path)[0])
        data_directory = os.path.dirname(mat_file_path)
        np.save(os.path.join(data_directory, 'images_' + basename) + '.npy', images)
        np.save(os.path.join(data_directory, 'depths_' + basename) + '.npy', depths)

    @staticmethod
    def convert_mat_data_to_numpy_array(mat_data, variable_name_in_mat_data, number_of_samples=None):
        """
        Converts a mat data variable to a numpy array.

        :param mat_data: The mat data containing the variable to be converted.
        :type mat_data: h5py.File
        :param variable_name_in_mat_data: The name of the variable to extract.
        :type variable_name_in_mat_data: str
        :param number_of_samples: The number of samples to extract.
        :type number_of_samples: int
        :return: The numpy array.
        :rtype: np.ndarray
        """
        mat_variable = mat_data.get(variable_name_in_mat_data)
        reversed_array = np.array(mat_variable)
        array = reversed_array.transpose()
        if variable_name_in_mat_data in ('images', 'depths'):
            array = np.rollaxis(array, -1)
        return array[:number_of_samples]

    @staticmethod
    def crop_data(array):
        """
        Crop the NYU data to remove dataless borders.

        :param array: The numpy array to crop
        :type array: np.ndarray
        :return: The cropped data.
        :rtype: np.ndarray
        """
        return array[:, 8:-8, 8:-8]

    def convert_numpy_to_tfrecords(self, images, labels=None):
        """
        Converts numpy arrays to a TFRecords.
        """
        number_of_examples = images.shape[0]
        if labels is not None:
            if labels.shape[0] != number_of_examples:
                raise ValueError("Images count %d does not match label count %d." %
                                 (labels.shape[0], number_of_examples))
            label_height = labels.shape[1]
            if len(labels.shape) > 2:
                label_width = labels.shape[2]
            else:
                label_width = 1
            if len(labels.shape) > 3:
                label_depth = labels.shape[3]
            else:
                label_depth = 1
        else:
            label_height, label_width, label_depth = None, None, None  # Line to quiet inspections
        image_height = images.shape[1]
        image_width = images.shape[2]
        image_depth = images.shape[3]

        filename = os.path.join(self.data_directory, self.data_name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(number_of_examples):
            image_raw = images[index].tostring()
            features = {
                'image_height': _int64_feature(image_height),
                'image_width': _int64_feature(image_width),
                'image_depth': _int64_feature(image_depth),
                'image_raw': _bytes_feature(image_raw),
            }
            if labels is not None:
                label_raw = labels[index].tostring()
                features.update({
                    'label_height': _int64_feature(label_height),
                    'label_width': _int64_feature(label_width),
                    'label_depth': _int64_feature(label_depth),
                    'label_raw': _bytes_feature(label_raw)
                })
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())

    def import_mat_file(self, mat_path):
        """
        Imports a Matlab mat file into the data images and labels (concatenating the arrays if they already exists).

        :param mat_path: The path to the mat file to import.
        :type mat_path: str
        """
        with h5py.File(mat_path, 'r') as mat_data:
            uncropped_images = self.convert_mat_data_to_numpy_array(mat_data, 'images')
            images = self.crop_data(uncropped_images)
            uncropped_labels = self.convert_mat_data_to_numpy_array(mat_data, 'depths')
            labels = self.crop_data(uncropped_labels)
            self.images = images
            self.labels = labels

    def import_file(self, file_path):
        """
        Import the data.
        Should be overwritten by subclasses.

        :param file_path: The file path of the file to be imported.
        :type file_path: str
        """
        self.import_mat_file(file_path)

    def convert_to_tfrecords(self):
        """
        Converts the data to a TFRecords file.
        """
        self.convert_numpy_to_tfrecords(self.images, self.labels)

    def generate_all_tfrecords(self):
        """
        Creates the TFRecords for the data.
        """
        import_file_paths = self.attain_import_file_paths()
        if not import_file_paths:
            print('No data found in %s.' % self.import_directory)
        for import_file_path in import_file_paths:
            print('Converting %s...' % str(import_file_path))
            self.import_file(import_file_path)
            self.obtain_export_name(import_file_path)
            self.convert_to_tfrecords()

    def obtain_export_name(self, import_file_path):
        """
        Extracts the name to be used for the export file.

        :param import_file_path: The import path.
        :type import_file_path: str | (str, str)
        :return: The name of the export file.
        :rtype: str
        """
        self.data_name = os.path.splitext(os.path.basename(import_file_path))[0]

    def attain_import_file_paths(self):
        """
        Gets a list of all the file paths for files to be imported.

        :return: The list of the file paths to be imported.
        :rtype: list[str]
        """
        import_file_paths = []
        for file_directory, _, file_names in os.walk(self.import_directory):
            mat_names = [file_name for file_name in file_names if file_name.endswith('.mat')]
            for mat_name in mat_names:
                mat_path = os.path.abspath(os.path.join(file_directory, mat_name))
                import_file_paths.append(mat_path)
        return import_file_paths


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    data = GoData()
    data.generate_all_tfrecords()
