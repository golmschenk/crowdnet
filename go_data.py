"""
Code for managing the TFRecord data.
"""
import os
import h5py
import numpy as np
import tensorflow as tf


class GoData:
    """
    A class for managing the TFRecord data.
    """

    def __init__(self):
        self.data_directory = 'data'
        self.data_name = 'nyud_micro'
        self.import_directory = 'data/import'
        self.height = 464 // 8
        self.width = 624 // 8
        self.channels = 3
        self.original_height = 464
        self.original_width = 624
        self.image_shape = [self.height, self.width, self.channels]
        self.label_shape = [self.height, self.width, 1]
        self.images = None
        self.labels = None

        self.train_size = 9
        self.validation_size = 1
        self.test_size = 0

        os.nice(10)

    @property
    def data_path(self):
        """
        Gives the path to the data file.

        :return: The path to the data file.
        :rtype: str
        """
        return os.path.join(self.data_directory, self.data_name)

    def read_and_decode(self, filename_queue):
        """
        A definition of how TF should read a single example proto from the file record.

        :param filename_queue: The file name queue to be read.
        :type filename_queue: tf.QueueBase
        :return: The read file data including the image data and label data.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_raw': tf.FixedLenFeature([], tf.string),
            })

        flat_image = tf.decode_raw(features['image_raw'], tf.uint8)
        unnormalized_image = tf.reshape(flat_image, self.image_shape)
        image = tf.cast(unnormalized_image, tf.float32) * (1. / 255) - 0.5

        flat_label = tf.decode_raw(features['label_raw'], tf.float32)
        label = tf.reshape(flat_label, self.label_shape)

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
        if data_type:
            file_name = self.data_name + '.' + data_type + '.tfrecords'
        else:
            file_name = self.data_name + '.tfrecords'
        file_path = os.path.join(self.data_directory, file_name)

        with tf.name_scope('Input'):
            file_name_queue = tf.train.string_input_producer([file_path], num_epochs=num_epochs)

            image, label = self.read_and_decode(file_name_queue)

            images, labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=2,
                capacity=500 + 3 * batch_size, min_after_dequeue=500
            )

            return images, labels

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

    def numpy_files_to_tfrecords(self, augment=False):
        """
        Converts NumPy files to a TFRecords file.
        """
        self.load_numpy_files()
        self.shrink()
        if augment:
            self.augment_data_set()
        self.convert_to_tfrecords()

    def load_numpy_files(self):
        """
        Loads data from the numpy files into the object.
        """
        images_numpy_file_path = os.path.join(self.data_path + '_images.npy')
        labels_numpy_file_path = os.path.join(self.data_path + '_labels.npy')
        self.images = np.load(images_numpy_file_path)
        self.labels = np.load(labels_numpy_file_path)

    def convert_numpy_to_tfrecords(self, images, labels, data_set='train'):
        """
        Converts numpy arrays to a TFRecords.
        """
        number_of_examples = labels.shape[0]
        if images.shape[0] != number_of_examples:
            raise ValueError("Images count %d does not match label count %d." %
                             (images.shape[0], number_of_examples))
        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]

        filename = os.path.join(self.data_directory, self.data_name + '.' + data_set + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(number_of_examples):
            image_raw = images[index].tostring()
            label_raw = labels[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'channels': _int64_feature(depth),
                'image_raw': _bytes_feature(image_raw),
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
        if array.shape[1] == self.height and array.shape[2] == self.width:
            return array  # The shape is already right, so don't needlessly process.
        compression_shape = [
            array.shape[0],
            self.height,
            array.shape[1] // self.height,
            self.width,
            array.shape[2] // self.width,
        ]
        if len(array.shape) == 4:
            compression_shape.append(self.channels)
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

    def import_recursive_mat_directory(self):
        """
        Imports the import directory of Matlab mat files recursively into the data images and labels.
        """
        for file_directory, _, file_names in os.walk(self.import_directory):
            mat_names = [file_name for file_name in file_names if file_name.endswith('.mat')]
            for mat_name in mat_names:
                print('Importing %s' % mat_name)
                self.import_mat_file(os.path.join(file_directory, mat_name))
                assert self.images.shape[0] == self.labels.shape[0]

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
            if self.images is None:
                self.images = images
                self.labels = labels
            else:
                self.images = np.concatenate((self.images, images))
                self.labels = np.concatenate((self.labels, labels))

    def import_data(self):
        """
        Import the data.
        Should be overwritten by subclasses.
        """
        self.import_recursive_mat_directory()

    def preprocess(self):
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
        Converts the data to train, validation, and test TFRecords. If the size of any dataset is not specified, the
        remaining data is placed into that dataset. For example, if there are 300 samples, and train_size is 200 with
        neither of the other sizes yet, then the training dataset will contain 200 samples, the validation dataset
        will contain 100 samples, and no test set will be created.
        """
        used = 0
        for name, size in [('train', self.train_size), ('validation', self.validation_size), ('test', self.test_size)]:
            if size in ('all', 'remaining', 'rest'):
                self.convert_numpy_to_tfrecords(self.images[used:], self.labels[used:], data_set=name)
                return
            elif size:
                self.convert_numpy_to_tfrecords(self.images[used:used+size], self.labels[used:used+size], data_set=name)
                used += size
                if used >= len(self.labels):
                    return

    def generate_tfrecords(self):
        """
        Creates the TFRecords for the data.
        """
        print('Importing the data...')
        self.import_data()
        print('Preprocessing the data...')
        self.preprocess()
        print('Generating the TF records...')
        self.convert_to_tfrecords()
        print('Done.')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    data = GoData()
    data.generate_tfrecords()
