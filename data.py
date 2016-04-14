"""
Code for managing the TFRecord data.
"""
import os
import h5py
import numpy as np
import tensorflow as tf
import cv2


class Data:
    """
    A class for managing the TFRecord data.
    """

    def __init__(self, data_directory='examples', data_name='nyud'):
        self.data_directory = data_directory
        self.data_name = data_name
        self.height = 464 // 8
        self.width = 624 // 8
        self.channels = 3
        self.original_height = 464
        self.original_width = 624

    def read_and_decode(self, filename_queue):
        """
        A definition of how TF should read the file record.
        Slightly altered version from https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/how_tos/ \
                                      reading_data/fully_connected_reader.py

        :param filename_queue: The file name queue to be read.
        :type filename_queue: tf.QueueBase
        :return: The read file data including the image data and depth data.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'depth_raw': tf.FixedLenFeature([], tf.string),
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [self.height, self.width, self.channels])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        depth = tf.decode_raw(features['depth_raw'], tf.float32)
        depth = tf.reshape(depth, [self.height, self.width, 1])

        return image, depth

    def inputs(self, data_type, batch_size, num_epochs=None):
        """
        Prepares the data inputs.
        Slightly altered version from https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/how_tos/ \
                                      reading_data/fully_connected_reader.py

        :param data_type: The type of data file (usually train, validation, or test).
        :type data_type: str
        :param batch_size: The size of the batches
        :type batch_size: int
        :param num_epochs: Number of epochs to run for. Infinite if None.
        :type num_epochs: int | None
        :return: The images and depths inputs.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        file_path = os.path.join(self.data_directory, self.data_name + '.' + data_type + '.tfrecords')

        with tf.name_scope('Input'):
            filename_queue = tf.train.string_input_producer([file_path], num_epochs=num_epochs)

            image, depth = self.read_and_decode(filename_queue)

            images, depths = tf.train.shuffle_batch(
                [image, depth], batch_size=batch_size, num_threads=2,
                capacity=500 + 3 * batch_size, min_after_dequeue=500
            )

            return images, depths

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
        untransposed_array = np.array(mat_variable)
        if number_of_samples:
            untransposed_array = untransposed_array[:number_of_samples]
        if untransposed_array.ndim == 3:  # For depth images.
            return untransposed_array.transpose((0, 2, 1))
        else:  # For RGB images.
            return untransposed_array.transpose((0, 3, 2, 1))

    @staticmethod
    def crop_data(data):
        """
        Crop the NYU data to remove dataless borders.

        :param data: The numpy array to crop
        :type data: np.ndarray
        :return: The cropped data.
        :rtype: np.ndarray
        """
        return data[:, 8:-8, 8:-8]

    def convert_mat_to_tfrecord(self, mat_file_path, data_directory='examples'):
        """
        Converts the mat file data into a TFRecords file.

        :param mat_file_path: The path to mat file to convert.
        :type mat_file_path: str
        :param data_directory: Currently unused.
        :type data_directory: str
        """
        mat_data = h5py.File(mat_file_path, 'r')
        uncropped_images = self.convert_mat_data_to_numpy_array(mat_data, 'images')
        images = self.crop_data(uncropped_images)
        images = self.rebin(images, self.height, self.width)
        uncropped_depths = self.convert_mat_data_to_numpy_array(mat_data, 'depths')
        depths = self.crop_data(uncropped_depths)
        depths = self.rebin(depths, self.height, self.width)
        basename = os.path.basename(os.path.splitext(mat_file_path)[0])
        data_directory = os.path.dirname(mat_file_path)
        self.convert_to_tfrecord(images, depths, basename + '.train', data_directory)

    @staticmethod
    def convert_to_tfrecord(images, depths, name, data_directory):
        """
        Converts the data to a TFRecord.

        :param images: The images to be converted.m
        :type images: np.ndarray
        :param depths: The depths to be converted.
        :type depths: np.ndarray
        :param name: The name of the file to be saved (usually the type [i.e. train, validation, or test]).
        :type name: str
        :param data_directory: The directory of in which to save the data.
        :type data_directory: str
        """
        number_of_examples = depths.shape[0]
        if images.shape[0] != number_of_examples:
            raise ValueError("Images size %d does not match label size %d." %
                             (images.shape[0], number_of_examples))
        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]

        filename = os.path.join(data_directory, name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(number_of_examples):
            image_raw = images[index].tostring()
            depth_raw = depths[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'channels': _int64_feature(depth),
                'image_raw': _bytes_feature(image_raw),
                'depth_raw': _bytes_feature(depth_raw),
            }))
            writer.write(example.SerializeToString())

    def rebin(self, array, height, width):
        """
        Rebins the NumPy array into a new size, averaging the bins between.

        :param array: The array to resize.
        :type array: np.ndarray
        :param height: The height to rebin to.
        :type height: int
        :param width: The width to rebin to.
        :type width: int
        :return: The resized array.
        :rtype: np.ndarray
        """
        compression_shape = [
            array.shape[0],
            height,
            array.shape[1] // height,
            width,
            array.shape[2] // width,
        ]
        if len(array.shape) == 4:
            compression_shape.append(self.channels)
            return array.reshape(compression_shape).mean(4).mean(2).astype(np.uint8)
        else:
            return array.reshape(compression_shape).mean(4).mean(2)

    @staticmethod
    def gaussian_noise_augmentation(images):
        """
        Applies random gaussian noise to a set of images.

        :param images: The images to add the noise to.
        :type images: np.ndarray
        :return: The noisy images.
        :rtype: np.ndarray
        """
        noise = np.zeros(images.shape).astype(np.int32)
        cv2.randn(noise, 0, 5)
        return (images.astype(np.int32) + noise).clip(0, 255).astype(np.uint8)

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
            offset_array[offset:] = offset_array[:offset]
        else:
            offset_array[:offset] = offset_array[offset:]
        offset_array = np.swapaxes(offset_array, 0, axis)
        return offset_array

    def augment_dataset(self, images, depths):
        offset_images0 = self.offset_array(images, 1, 1)
        offset_depths0 = self.offset_array(depths, 1, 1)
        offset_images1 = self.offset_array(images, -1, 1)
        offset_depths1 = self.offset_array(depths, -1, 1)
        offset_images2 = self.offset_array(images, 1, 2)
        offset_depths2 = self.offset_array(depths, 1, 2)
        offset_images3 = self.offset_array(images, -1, 2)
        offset_depths3 = self.offset_array(depths, -1, 2)
        images = np.concatenate((images, offset_images0, offset_images1, offset_images2, offset_images3))
        depths = np.concatenate((depths, offset_depths0, offset_depths1, offset_depths2, offset_depths3))

        noisy_images0 = self.gaussian_noise_augmentation(images)
        noisy_images1 = self.gaussian_noise_augmentation(images)
        noisy_images2 = self.gaussian_noise_augmentation(images)
        noisy_images3 = self.gaussian_noise_augmentation(images)
        images = np.concatenate((images, noisy_images0, noisy_images1, noisy_images2, noisy_images3))
        depths = np.concatenate((depths, depths, depths, depths, depths))
        return images, depths


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    os.nice(10)

    data = Data()
    data.convert_mat_to_tfrecord('examples/nyud_micro.mat')