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
        self.dataset_container = 'directory'

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
    def read_and_decode_single_example_from_tfrecords(file_name_queue):
        """
        A definition of how TF should read a single example proto from the file record.

        :param file_name_queue: The file name queue to be read.
        :type file_name_queue: tf.QueueBase
        :return: The read file data including the image data and label data.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        go_tfrecords_reader = GoTFRecordsReader(file_name_queue)
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

    def create_input_tensors_for_dataset(self, data_type, batch_size, num_epochs=None):
        """
        Prepares the data inputs.

        :param data_type: The type of data file (usually train, validation, or test).
        :type data_type: str
        :param batch_size: The size of the batches
        :type batch_size: int
        :param num_epochs: Number of epochs to run for. Infinite if None.
        :type num_epochs: int or None
        :return: The images and depths inputs.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        if self.dataset_container == 'file':
            file_name_queue = self.file_name_queue_for_dataset_file(data_type, num_epochs)
        else:
            file_name_queue = self.file_name_queue_for_dataset_directory(data_type, num_epochs)

        image, label = self.read_and_decode_single_example_from_tfrecords(file_name_queue)
        image, label = self.preaugmentation_preprocess(image, label)
        if data_type == 'train':
            image, label = self.augment(image, label)
        image, label = self.postaugmentation_preprocess(image, label)

        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=500 + 3 * batch_size, min_after_dequeue=500
        )

        return images, labels

    def file_name_queue_for_dataset_file(self, data_type=None, num_epochs=None):
        """
        Creates the files name queue for a single TFRecords file.

        :param data_type: The type of dataset being created.
        :type data_type: str
        :param num_epochs: Number of epochs to run for. Infinite if None.
        :type num_epochs: int or None
        :return: The file name queue.
        :rtype: tf.QueueBase
        """
        if data_type:
            file_name = self.data_name + '.' + data_type + '.tfrecords'
        else:
            file_name = self.data_name + '.tfrecords'
        file_path = os.path.join(self.data_directory, file_name)
        file_name_queue = tf.train.string_input_producer([file_path], num_epochs=num_epochs)
        return file_name_queue

    def file_name_queue_for_dataset_directory(self, data_type, num_epochs=None):
        """
        Creates the files name queue for a single TFRecords file.

        :param data_type: The type of dataset being created.
        :type data_type: str
        :param num_epochs: Number of epochs to run for. Infinite if None.
        :type num_epochs: int or None
        :return: The file name queue.
        :rtype: tf.QueueBase
        """
        file_paths = []
        for file_path in glob.glob(os.path.join(self.data_directory, data_type, '*.tfrecords')):
            file_paths.append(file_path)
        file_name_queue = tf.train.string_input_producer(file_paths, num_epochs=num_epochs)
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

    def numpy_files_to_tfrecords(self):
        """
        Converts NumPy files to a TFRecords file.
        """
        self.load_numpy_files()
        self.convert_to_tfrecords()

    def load_numpy_files(self):
        """
        Loads data from the numpy files into the object.
        """
        images_numpy_file_path = os.path.join(self.data_path + '_images.npy')
        labels_numpy_file_path = os.path.join(self.data_path + '_labels.npy')
        self.images = np.load(images_numpy_file_path)
        self.labels = np.load(labels_numpy_file_path)
        if self.labels.dtype == np.float64:
            self.labels = self.labels.astype(np.float32)

    def convert_numpy_to_tfrecords(self, images, labels):
        """
        Converts numpy arrays to a TFRecords.
        """
        number_of_examples = labels.shape[0]
        if images.shape[0] != number_of_examples:
            raise ValueError("Images count %d does not match label count %d." %
                             (images.shape[0], number_of_examples))
        image_height = images.shape[1]
        image_width = images.shape[2]
        image_depth = images.shape[3]
        label_height = labels.shape[1]
        if len(labels.shape) > 2:
            label_width = labels.shape[2]
        else:
            label_width = 1
        if len(labels.shape) > 3:
            label_depth = labels.shape[3]
        else:
            label_depth = 1

        filename = os.path.join(self.data_directory, self.data_name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(number_of_examples):
            image_raw = images[index].tostring()
            label_raw = labels[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_height': _int64_feature(image_height),
                'image_width': _int64_feature(image_width),
                'image_depth': _int64_feature(image_depth),
                'image_raw': _bytes_feature(image_raw),
                'label_height': _int64_feature(label_height),
                'label_width': _int64_feature(label_width),
                'label_depth': _int64_feature(label_depth),
                'label_raw': _bytes_feature(label_raw),
            }))
            writer.write(example.SerializeToString())

    def shrink(self):
        """
        Rebins the data arrays into the specified data size.
        """
        self.images = self.shrink_array_with_rebinning(self.images)
        self.labels = self.shrink_array_with_rebinning(self.labels)

    def shrink_array_with_rebinning(self, array):
        """
        Rebins the NumPy array into a new size, averaging the bins between.
        :param array: The array to resize.
        :type array: np.ndarray
        :return: The resized array.
        :rtype: np.ndarray
        """
        if array.shape[1] == self.image_height and array.shape[2] == self.image_width:
            return array  # The shape is already right, so don't needlessly process.
        compression_shape = [
            array.shape[0],
            self.image_height,
            array.shape[1] // self.image_height,
            self.image_width,
            array.shape[2] // self.image_width,
        ]
        if len(array.shape) == 4:
            compression_shape.append(self.image_depth)
            return array.reshape(compression_shape).mean(4).mean(2).astype(np.uint8)
        else:
            return array.reshape(compression_shape).mean(4).mean(2)

    def gaussian_noise_augmentation(self, standard_deviation, number_of_variations):
        """
        Applies random gaussian noise to the images.

        :param standard_deviation: The standard deviation of the gaussian noise.
        :type standard_deviation: float
        :param number_of_variations: The number of noisy copies to create.
        :type number_of_variations: int
        """
        augmented_images_list = [self.images]
        augmented_labels_list = [self.labels]
        for _ in range(number_of_variations):
            # noinspection PyTypeChecker
            noise = np.random.normal(np.zeros(shape=self.image_shape, dtype=np.int16), standard_deviation)
            augmented_images_list.append((self.images.astype(np.int16) + noise).clip(0, 255).astype(np.uint8))
            augmented_labels_list.append(self.labels)
        self.images = np.concatenate(augmented_images_list)
        self.labels = np.concatenate(augmented_labels_list)

    @staticmethod
    def offset_array(array, offset, axis):
        """
        Offsets an array by the given amount (simply by copying the array to the given portion).
        Note, this is only working for very specific cases at the moment.

        :param array: The array to offset.
        :type array: np.ndarray
        :param offset: The amount of the offset.
        :type offset: int
        :param axis: The axis to preform the offset on.
        :type axis: int
        :return: The offset array.
        :rtype: np.ndarray
        """
        offset_array = np.copy(array)
        offset_array = np.swapaxes(offset_array, 0, axis)
        if offset > 0:
            offset_array[offset:] = offset_array[:-offset]
        else:
            offset_array[:offset] = offset_array[-offset:]
        offset_array = np.swapaxes(offset_array, 0, axis)
        return offset_array

    def offset_augmentation(self, offset_limit):
        """
        Augments the data using a crude spatial shifting based on a given offset.

        :param offset_limit: The value of the maximum offset.
        :type offset_limit: int
        """
        augmented_images_list = [self.images]
        augmented_labels_list = [self.labels]
        for axis in [1, 2]:
            for offset in range(-offset_limit, offset_limit + 1):
                if offset == 0:
                    continue
                augmented_images_list.append(self.offset_array(self.images, offset, axis))
                augmented_labels_list.append(self.offset_array(self.labels, offset, axis))
        self.images = np.concatenate(augmented_images_list)
        self.labels = np.concatenate(augmented_labels_list)

    def augment_data_set(self):
        """
        Augments the data set with some basic approaches
        """
        print('Augmenting with spatial jittering...')
        self.offset_augmentation(1)
        print('Augmenting with gaussian noise...')
        self.gaussian_noise_augmentation(10, 4)

    def shuffle(self):
        """
        Shuffles the images and labels together.
        """
        permuted_indexes = np.random.permutation(len(self.images))
        self.images = self.images[permuted_indexes]
        self.labels = self.labels[permuted_indexes]

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

    def pretfrecords_preprocess(self):
        """
        Preprocesses the data.
        Should be overwritten by subclasses.
        """
        print('Shrinking the data...')
        self.shrink()
        print('Augmenting the data...')
        self.augment_data_set()
        print('Shuffling the data...')
        self.shuffle()

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
