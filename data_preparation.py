"""
Code for preparing data.
"""
import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


class DataPreparation:
    """
    A class for data preparation.
    """

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

    def convert_mat_to_tfrecord(self, mat_file_path, data_directory='data'):
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
        uncropped_depths = self.convert_mat_data_to_numpy_array(mat_data, 'depths')
        depths = self.crop_data(uncropped_depths)
        basename = os.path.basename(os.path.splitext(mat_file_path)[0])
        data_directory = os.path.dirname(mat_file_path)
        self.convert_to_tfrecord(images, depths, basename + '.train', data_directory)

    @staticmethod
    def convert_to_tfrecord(images, depths, name, data_directory):
        """
        Converts the data to a TFRecord.

        :param images: The images to be converted.
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


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    data_preparation = DataPreparation()
    # data_preparation.convert_mat_file_to_numpy_file('data/nyu_depth_v2_labeled.mat', number_of_samples=10)
