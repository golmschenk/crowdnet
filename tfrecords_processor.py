"""
Code for dealing with reading and interacting with TFRecords outside of the main network.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError


class TFRecordsProcessor:
    """
    A class for dealing with reading and interacting with TFRecords outside of the main network.
    """

    def read_to_numpy(self, file_name, data_type=None):
        """
        Reads entire TFRecords file as NumPy.

        :param file_name: The TFRecords file name to read.
        :type file_name: str
        :param data_type: Data type of data. Used if that data type doesn't include things like labels.
        :type data_type: str
        :return: The images and labels NumPy
        :rtype: (np.ndarray, np.ndarray)
        """
        feature_types = self.attain_feature_types(data_type)
        images = []
        labels = []
        for tfrecord in tf.python_io.tf_record_iterator(file_name):
            with tf.Graph().as_default() as graph:  # Create a separate as this runs slow when on one graph.
                features = tf.parse_single_example(tfrecord, features=feature_types)
                image_shape, label_shape = self.extract_shapes_from_tfrecords_features(features, data_type)
                flat_image = tf.decode_raw(features['image_raw'], tf.uint8)
                image_tensor = tf.reshape(flat_image, image_shape)
                image_tensor = tf.squeeze(image_tensor)
                if data_type != 'deploy':
                    flat_label = tf.decode_raw(features['label_raw'], tf.float32)
                    label_tensor = tf.reshape(flat_label, label_shape)
                    label_tensor = tf.squeeze(label_tensor)
                else:
                    label_tensor = tf.constant(-1.0, dtype=tf.float32, shape=[1, 1, 1])
                with tf.Session(graph=graph) as session:
                    initialize_op = tf.global_variables_initializer()
                    session.run(initialize_op)
                    image, label = session.run([image_tensor, label_tensor])
            images.append(image)
            labels.append(label)
        return np.stack(images), np.stack(labels)

    @staticmethod
    def attain_feature_types(data_type):
        """
        Get the needed features type dictionary to read the TFRecords.

        :param data_type: The type of data being process. Determines whether to look for labels.
        :type data_type: str
        :return: The feature type dictionary.
        :rtype: dict[str, tf.FixedLenFeature]
        """
        feature_types = {
            'image_height': tf.FixedLenFeature([], tf.int64),
            'image_width': tf.FixedLenFeature([], tf.int64),
            'image_depth': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
        if data_type != 'deploy':
            feature_types.update({
                'label_height': tf.FixedLenFeature([], tf.int64),
                'label_width': tf.FixedLenFeature([], tf.int64),
                'label_depth': tf.FixedLenFeature([], tf.int64),
                'label_raw': tf.FixedLenFeature([], tf.string),
            })
        if data_type == 'unlabeled':
            feature_types.update({
                'label_guess': tf.FixedLenFeature([], tf.float32)
            })
        return feature_types

    def create_image_and_label_inputs_from_file_name_queue(self, file_name_queue, data_type=None):
        """
        Creates the inputs for the image and label for a given file name queue.

        :param file_name_queue: The file name queue to be used.
        :type file_name_queue: tf.Queue
        :param data_type: The type of data (train, validation, test, deploy, etc) to determine how to process.
        :type data_type: str
        :return: The image and label inputs.
        :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_name_queue)
        feature_types = self.attain_feature_types(data_type)
        features = tf.parse_single_example(serialized_example, features=feature_types)

        image_shape, label_shape = self.extract_shapes_from_tfrecords_features(features, data_type)

        flat_image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(flat_image, image_shape)

        if data_type != 'deploy':
            flat_label = tf.decode_raw(features['label_raw'], tf.float32)
            label = tf.reshape(flat_label, label_shape)
        else:
            # Makes a fake label tensor for preprocessing to work on.
            label = tf.constant(-1.0, dtype=tf.float32, shape=[1, 1, 1])
        if data_type == 'unlabeled':
            label_guess = tf.cast(features['label_guess'], tf.float32)
        else:
            label_guess = -1
        return image, label, label_guess

    @staticmethod
    def extract_shapes_from_tfrecords_features(features, data_type):
        """
        Extracts the image and label shapes from the TFRecords' features. Uses a short TF session to do so.

        :param features: The recovered TFRecords' protobuf features.
        :type features: dict[str, tf.Tensor]
        :param data_type: The type of data (train, validation, test, deploy, etc) to determine how to process.
        :type data_type: str
        :return: The image and label shape tuples.
        :rtype: (int, int, int), (int, int, int)
        """
        image_height_tensor = tf.cast(features['image_height'], tf.int64)
        image_width_tensor = tf.cast(features['image_width'], tf.int64)
        image_depth_tensor = tf.cast(features['image_depth'], tf.int64)
        if data_type == 'deploy':
            label_height_tensor, label_width_tensor, label_depth_tensor = None, None, None  # Line to quiet inspections
        else:
            label_height_tensor = tf.cast(features['label_height'], tf.int64)
            label_width_tensor = tf.cast(features['label_width'], tf.int64)
            label_depth_tensor = tf.cast(features['label_depth'], tf.int64)
        # To read the TFRecords file, we need to start a TF session (including queues to read the file name).
        with tf.Session() as session:
            initialize_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            session.run(initialize_op)
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coordinator)
            if data_type == 'deploy':
                image_height, image_width, image_depth = session.run(
                    [image_height_tensor, image_width_tensor, image_depth_tensor])
                label_shape = None
            else:
                image_height, image_width, image_depth, label_height, label_width, label_depth = session.run(
                    [image_height_tensor, image_width_tensor, image_depth_tensor, label_height_tensor,
                     label_width_tensor, label_depth_tensor])
                label_shape = (label_height, label_width, label_depth)
            coordinator.request_stop()
            coordinator.join(threads)
        image_shape = (image_height, image_width, image_depth)
        return image_shape, label_shape

    @staticmethod
    def write_from_numpy(file_name, image_shape, images, label_shape, labels, label_guess=None):
        """
        Write a TFRecords from NumPy.

        :param file_name: The file name to write to.
        :type file_name: str
        :param image_shape: The size of each image.
        :type image_shape: list[int]
        :param images: The NumPy array of images.
        :type images: np.ndarray
        :param label_shape: The size of each label.
        :type label_shape: list[int]
        :param labels: The NumPy array of labels.
        :type labels: np.ndarray
        """
        writer = tf.python_io.TFRecordWriter(file_name)
        if labels is not None:
            label_shape += (1,) * (3 - len(label_shape))  # Always expand the labels shape to 3D.
        for index in range(images.shape[0]):
            image_raw = images[index].tostring()
            features = {
                'image_height': _int64_feature(image_shape[0]),
                'image_width': _int64_feature(image_shape[1]),
                'image_depth': _int64_feature(image_shape[2]),
                'image_raw': _bytes_feature(image_raw),
            }
            if labels is not None:
                label_raw = labels[index].tostring()
                features.update({
                    'label_height': _int64_feature(label_shape[0]),
                    'label_width': _int64_feature(label_shape[1]),
                    'label_depth': _int64_feature(label_shape[2]),
                    'label_raw': _bytes_feature(label_raw),
                })
            if label_guess:
                features.update({
                    'label_guess': _float_feature(label_guess)
                })
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())

    def split_tfrecords(self, file_name, number_of_parts=2, delete_original=False):
        """
        Split the TFRecords into multiple parts of equal size.

        :param file_name: The file name to split.
        :type file_name: str
        :param number_of_parts: The number of parts to split the file into.
        :type number_of_parts: int
        :param delete_original: Whether or not to delete the original file.
        :type delete_original: bool
        """
        images, labels = self.read_to_numpy(file_name)
        images_arrays = np.array_split(images, number_of_parts)
        labels_arrays = np.array_split(labels, number_of_parts)
        for index in range(len(images_arrays)):
            file_name_without_extension, extension = os.path.splitext(file_name)
            part_file_name = '{}_{}{}'.format(file_name_without_extension, index, extension)
            self.write_from_numpy(part_file_name, images.shape[1:], images_arrays[index], labels.shape[1:],
                                  labels_arrays[index])
        if delete_original:
            os.remove(file_name)

    def quadrantize_tfrecords(self, file_name):
        """
        Split the TFRecords into 4 parts spatial by quadrant.

        :param file_name: The file name to split.
        :type file_name: str
        """
        images, labels = self.read_to_numpy(file_name)
        images_top, images_bottom = np.array_split(images, 2, axis=1)
        labels_top, labels_bottom = np.array_split(labels, 2, axis=1)
        images_quadrant_2, images_quadrant_1 = np.array_split(images_top, 2, axis=2)
        images_quadrant_3, images_quadrant_4 = np.array_split(images_bottom, 2, axis=2)
        labels_quadrant_2, labels_quadrant_1 = np.array_split(labels_top, 2, axis=2)
        labels_quadrant_3, labels_quadrant_4 = np.array_split(labels_bottom, 2, axis=2)
        file_name_without_extension, extension = os.path.splitext(file_name)
        quadrant_1_file_name = '{}_{}{}'.format(file_name_without_extension, 'quadrant_1', extension)
        self.write_from_numpy(quadrant_1_file_name, images_quadrant_1.shape[1:], images_quadrant_1,
                              labels_quadrant_1.shape[1:], labels_quadrant_1)
        quadrant_2_file_name = '{}_{}{}'.format(file_name_without_extension, 'quadrant_2', extension)
        self.write_from_numpy(quadrant_2_file_name, images_quadrant_2.shape[1:], images_quadrant_2,
                              labels_quadrant_2.shape[1:], labels_quadrant_2)
        quadrant_3_file_name = '{}_{}{}'.format(file_name_without_extension, 'quadrant_3', extension)
        self.write_from_numpy(quadrant_3_file_name, images_quadrant_3.shape[1:], images_quadrant_3,
                              labels_quadrant_3.shape[1:], labels_quadrant_3)
        quadrant_4_file_name = '{}_{}{}'.format(file_name_without_extension, 'quadrant_4', extension)
        self.write_from_numpy(quadrant_4_file_name, images_quadrant_4.shape[1:], images_quadrant_4,
                              labels_quadrant_4.shape[1:], labels_quadrant_4)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
