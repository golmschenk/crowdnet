"""
Code related to the DepthNet.
"""

import tensorflow as tf
import time

from data import Data


class DepthNet:
    """
    The class to build and interact with the DepthNet TensorFlow graph.
    """

    def __init__(self):
        self.batch_size = 5
        self.number_of_epochs = 50000
        self.initial_learning_rate = 0.001
        self.summary_step_period = 1

    def inference(self, images):
        """
        Performs a forward pass estimating depth maps from RGB images.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The depth maps tensor.
        :rtype: tf.Tensor
        """
        with tf.name_scope('conv1'):
            w_conv1 = weight_variable([5, 5, 3, 8])
            b_conv1 = bias_variable([8])

            h_conv = self.leaky_relu(conv2d(images, w_conv1) + b_conv1)

        for index in range(50):
            with tf.name_scope('conv' + str(index + 2)):
                w_conv = weight_variable([5, 5, 8, 8])
                b_conv = bias_variable([8])

                h_conv = self.leaky_relu(conv2d(h_conv, w_conv) + b_conv)

        with tf.name_scope('conv52'):
            w_conv52 = weight_variable([5, 5, 8, 1])
            b_conv52 = bias_variable([1])

            predicted_depths = self.leaky_relu(conv2d(h_conv, w_conv52) + b_conv52)

        return predicted_depths

    @staticmethod
    def leaky_relu(x):
        return tf.maximum(0.0001 * x, x)

    @staticmethod
    def relative_differences(predicted_depths, depths):
        """
        Determines the absolute L1 relative differences between two depth maps.

        :param predicted_depths: The first depth map tensor (usually the predicted depths).
        :type predicted_depths: tf.Tensor
        :param depths: The second depth map tensor (usually the actual depths).
        :type depths: tf.Tensor
        :return: The difference tensor.
        :rtype: tf.Tensor
        """
        difference = tf.abs(predicted_depths - depths)
        return difference / depths

    def training(self, value_to_minimize):
        """
        Create and add the training op to the graph.

        :param value_to_minimize: The value to train on.
        :type value_to_minimize: tf.Tensor
        :return: The training op.
        :rtype: tf.Operation
        """
        return tf.train.AdamOptimizer(self.initial_learning_rate).minimize(value_to_minimize)

    @staticmethod
    def convert_to_heat_map_rgb(tensor):
        """
        Convert a tensor to a heat map.

        :param tensor: The tensor values to be converted.
        :type tensor: tf.Tensor
        :return: The heat map image tensor.
        :rtype: tf.Tensor
        """
        maximum = tf.reduce_max(tensor)
        minimum = tf.reduce_min(tensor)
        ratio = 2 * (tensor - minimum) / (maximum - minimum)
        b = tf.maximum(0.0, (1 - ratio))
        r = tf.maximum(0.0, (ratio - 1))
        g = 1 - b - r
        return tf.concat(3, [r, g, b]) - 0.5

    def side_by_side_image_summary(self, images, depths, predicted_depths, depth_differences):
        """
        Combines the image, depth, and difference tensors together into a presentable image. Then adds the
        image summary op to the graph.

        :param images: The original image.
        :type images: tf.Tensor
        :param depths: The tensor containing the actual depth values.
        :type depths: tf.Tensor
        :param predicted_depths: The tensor containing the predicted depths.
        :type predicted_depths: tf.Tensor
        :param depth_differences: The tensor containing the difference between the actual and predicted depths.
        :type depth_differences: tf.Tensor
        """
        depth_heat_map = self.convert_to_heat_map_rgb(depths)
        predicted_depth_heat_map = self.convert_to_heat_map_rgb(predicted_depths)
        depth_difference_heat_map = self.convert_to_heat_map_rgb(depth_differences)

        comparison_image = tf.concat(1, [images, depth_heat_map, predicted_depth_heat_map, depth_difference_heat_map])
        tf.image_summary('comparison', comparison_image)

    def train_network(self):
        """
        Adds the training operations and runs the training loop.
        """
        with tf.Graph().as_default():
            print('Preparing data...')
            # Setup the inputs.
            data = Data(data_directory='examples', data_name='nyud_micro')
            images, depths = data.inputs(data_type='train', batch_size=self.batch_size,
                                         num_epochs=self.number_of_epochs)

            print('Building graph...')
            # Add the forward pass operations to the graph.
            predicted_depths = self.inference(images)

            # Add the loss operations to the graph.
            with tf.name_scope('loss'):
                relative_differences = self.relative_differences(predicted_depths, depths)
                relative_difference_sum = tf.reduce_sum(relative_differences)
                tf.scalar_summary("Relative difference sum", relative_difference_sum)

            with tf.name_scope('comparison_summary'):
                self.side_by_side_image_summary(images, depths, predicted_depths, relative_differences)

            # Add the training operations to the graph.
            train_op = self.training(relative_difference_sum)

            # The op for initializing the variables.
            initialize_op = tf.initialize_all_variables()

            print('Starting training...')
            # Create a session for running operations in the Graph.
            session = tf.Session()

            # Prepare the summary operations.
            summaries_op = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter("/tmp/depth_net_logs", session.graph_def)

            # Initialize the variables.
            session.run(initialize_op)

            # Start input enqueue threads.
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

            # Preform the training loop.
            step = 0
            try:
                while not coordinator.should_stop():
                    # Regular training step.
                    start_time = time.time()
                    _, relative_difference_sum_value, summaries = session.run([train_op, relative_difference_sum,
                                                                               summaries_op])
                    duration = time.time() - start_time

                    # Information print and summary write step.
                    if step % self.summary_step_period == 0:
                        writer.add_summary(summaries, step)
                        print('Step %d: relative difference sum = %.2f (%.3f sec)' % (step,
                                                                                      relative_difference_sum_value,
                                                                                      duration))
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (self.number_of_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coordinator.request_stop()

            # Wait for threads to finish.
            coordinator.join(threads)
            session.close()


def weight_variable(shape):
    """
    Create a generic weight variable.

    :param shape: The shape of the weight variable.
    :type shape: list[int]
    :return: The weight variable.
    :rtype: tf.Variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    Create a generic bias variable.

    :param shape: The shape of the bias variable.
    :type shape: list[int]
    :return: The bias variable.
    :rtype: tf.Variable
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(images, weights):
    """
    Create a generic convolutional operation.

    :param images: The images to prefrom the convolution on.
    :type images: tf.Tensor
    :param weights: The weight variable to be applied.
    :type weights: tf.Variable
    :return: The convolutional operation.
    :rtype: tf.Tensor
    """
    return tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')


if __name__ == '__main__':
    depth_net = DepthNet()
    depth_net.train_network()
