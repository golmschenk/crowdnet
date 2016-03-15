"""
Code for managing the TFRecord data.
"""
import os
import tensorflow as tf


class Data:
    """
    A class for managing the TFRecord data.
    """

    def __init__(self, data_directory='examples', data_name='nyud'):
        self.data_directory = data_directory
        self.data_name = data_name
        self.height = 464
        self.width = 624
        self.channels = 3

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
        #image.set_shape([self.height, self.width, self.channels])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        depth = tf.decode_raw(features['depth_raw'], tf.float32)
        #depth.set_shape([self.height, self.width])
        depth = tf.reshape(depth, [self.height, self.width])

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
