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
        self.number_of_epochs = 10000
        self.initial_learning_rate = 0.01
        self.summary_step_period = 1

    def inference(self, images):
        with tf.name_scope('conv1') as scope:
            w_conv1 = weight_variable([5, 5, 3, 8])
            b_conv1 = bias_variable([8])

            h_conv1 = tf.nn.relu(conv2d(images, w_conv1) + b_conv1)

        with tf.name_scope('conv2') as scope:
            w_conv2 = weight_variable([5, 5, 8, 8])
            b_conv2 = bias_variable([8])

            h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2) + b_conv2)

        with tf.name_scope('conv3') as scope:
            w_conv3 = weight_variable([5, 5, 8, 8])
            b_conv3 = bias_variable([8])

            h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3) + b_conv3)

        with tf.name_scope('conv4') as scope:
            w_conv4 = weight_variable([5, 5, 8, 1])
            b_conv4 = bias_variable([1])

            predicted_depths_unsqueezed = tf.nn.relu(conv2d(h_conv3, w_conv4) + b_conv4)

        predicted_depths = tf.squeeze(predicted_depths_unsqueezed)

        return predicted_depths

    def relative_difference(self, predicted_depths, depths):
        difference = tf.abs(predicted_depths - depths)
        relative_differences = difference / depths
        return tf.reduce_sum(relative_differences)

    def training(self, value_to_minimize):
        return tf.train.AdamOptimizer(self.initial_learning_rate).minimize(value_to_minimize)

    def train_network(self):
        """
        Adds the training operations and runs the training loop.
        """
        with tf.Graph().as_default():
            # Setup the inputs.
            data = Data(data_directory='examples', data_name='nyud_micro')
            images, depths = data.inputs(data_type='train', batch_size=self.batch_size,
                                         num_epochs=self.number_of_epochs)

            # Add the forward pass operations to the graph.
            predicted_depths = self.inference(images)

            # Add the loss operations to the graph.
            with tf.name_scope('loss') as scope:
                relative_difference = self.relative_difference(predicted_depths, depths)
                relative_difference_summary = tf.scalar_summary("Relative difference sum", relative_difference)

            rgb_summary = tf.image_summary('RGB', images)
            depth_summary = tf.image_summary('Depth', tf.expand_dims(depths, -1))
            predicted_depth_summary = tf.image_summary('Predicted depth', tf.expand_dims(predicted_depths, -1))

            # Add the training operations to the graph.
            train_op = self.training(relative_difference)

            # The op for initializing the variables.
            initialize_op = tf.initialize_all_variables()

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
                    _, relative_difference_sum, summaries = session.run([train_op, relative_difference,
                                                                         summaries_op])
                    duration = time.time() - start_time

                    # Information print and summary write step.
                    if step % self.summary_step_period == 0:
                        writer.add_summary(summaries, step)
                        print('Step %d: relative difference sum = %.2f (%.3f sec)' % (step, relative_difference_sum,
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
