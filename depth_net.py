"""
Code related to the DepthNet.
"""
import datetime
import multiprocessing
import os
import time

import tensorflow as tf

from convenience import weight_variable, bias_variable, conv2d, leaky_relu, size_from_stride_two
from data import Data
from interface import Interface


class DepthNet(multiprocessing.Process):
    """
    The class to build and interact with the DepthNet TensorFlow graph.
    """

    def __init__(self, message_queue=None):
        super().__init__()
        self.batch_size = 8
        self.number_of_epochs = 50000
        self.initial_learning_rate = 0.00001
        self.data = Data(data_directory='examples', data_name='nyud_micro')
        self.queue = message_queue
        self.dropout_keep_probability = None

        self.summary_step_period = 1
        self.moving_average_loss = None
        self.moving_average_decay = 0.1
        self.log_directory = "logs"

    def inference(self, images):
        """
        Performs a forward pass estimating label maps from RGB images.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        with tf.name_scope('conv1'):
            w_conv = weight_variable([5, 5, 3, 32])
            b_conv = bias_variable([32])

            h_conv = leaky_relu(conv2d(images, w_conv) + b_conv)

        with tf.name_scope('conv2'):
            w_conv = weight_variable([5, 5, 32, 128])
            b_conv = bias_variable([128])

            h_conv = leaky_relu(conv2d(h_conv, w_conv) + b_conv)

        for index in range(9):
            with tf.name_scope('conv' + str(index + 3)):
                w_conv = weight_variable([5, 5, 128, 128])
                b_conv = bias_variable([128])

                h_conv = leaky_relu(conv2d(h_conv, w_conv) + b_conv)

        with tf.name_scope('conv12'):
            w_conv = weight_variable([5, 5, 128, 32])
            b_conv = bias_variable([32])

            h_conv = leaky_relu(conv2d(h_conv, w_conv) + b_conv)

        with tf.name_scope('fc1'):
            fc0_size = self.data.height * self.data.width * 32
            fc1_size = fc0_size // 4096
            h_fc = tf.reshape(h_conv, [-1, fc0_size])
            w_fc = weight_variable([fc0_size, fc1_size])
            b_fc = bias_variable([fc1_size])

            h_fc = leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)
            h_fc_drop = tf.nn.dropout(h_fc, self.dropout_keep_probability)

        with tf.name_scope('fc2'):
            fc2_size = fc1_size // 2
            w_fc = weight_variable([fc1_size, fc2_size])
            b_fc = bias_variable([fc2_size])

            h_fc = leaky_relu(tf.matmul(h_fc_drop, w_fc) + b_fc)
            h_fc_drop = tf.nn.dropout(h_fc, self.dropout_keep_probability)

        with tf.name_scope('fc3'):
            fc3_size = self.data.height * self.data.width
            w_fc = weight_variable([fc2_size, fc3_size])
            b_fc = bias_variable([fc3_size])

            h_fc = leaky_relu(tf.matmul(h_fc_drop, w_fc) + b_fc)
            predicted_labels = tf.reshape(h_fc, [-1, self.data.height, self.data.width, 1])

        return predicted_labels

    def standard_net_inference(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a AlexNet-like graph setup.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        with tf.name_scope('conv1'):
            w_conv = weight_variable([7, 7, 3, 16])
            b_conv = bias_variable([16])

            h_conv = leaky_relu(conv2d(images, w_conv) + b_conv)

        with tf.name_scope('conv2'):
            w_conv = weight_variable([7, 7, 16, 24])
            b_conv = bias_variable([24])

            h_conv = leaky_relu(conv2d(h_conv, w_conv, [1, 2, 2, 1]) + b_conv)

        with tf.name_scope('conv3'):
            w_conv = weight_variable([7, 7, 24, 32])
            b_conv = bias_variable([32])

            h_conv = leaky_relu(conv2d(h_conv, w_conv, [1, 2, 2, 1]) + b_conv)

        with tf.name_scope('fc1'):
            fc0_size = size_from_stride_two(self.data.height, iterations=2) * size_from_stride_two(self.data.width,
                                                                                                   iterations=2) * 32
            fc1_size = fc0_size // 2
            h_fc = tf.reshape(h_conv, [-1, fc0_size])
            w_fc = weight_variable([fc0_size, fc1_size])
            b_fc = bias_variable([fc1_size])

            h_fc = leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)

        with tf.name_scope('fc2'):
            fc2_size = fc1_size // 2
            w_fc = weight_variable([fc1_size, fc2_size])
            b_fc = bias_variable([fc2_size])

            h_fc = leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)

        with tf.name_scope('fc3'):
            fc3_size = self.data.height * self.data.width
            w_fc = weight_variable([fc2_size, fc3_size])
            b_fc = bias_variable([fc3_size])

            h_fc = leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)
            predicted_labels = tf.reshape(h_fc, [-1, self.data.height, self.data.width, 1])

        return predicted_labels

    def linear_classifier_inference(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using only a linear classifier.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        pixel_count = self.data.height * self.data.width
        flat_images = tf.reshape(images, [-1, pixel_count * self.data.channels])
        weights = weight_variable([pixel_count * self.data.channels, pixel_count], stddev=0.001)
        biases = bias_variable([pixel_count], constant=0.001)

        flat_predicted_labels = tf.matmul(flat_images, weights) + biases
        predicted_labels = tf.reshape(flat_predicted_labels, [-1, self.data.height, self.data.width, 1])
        return predicted_labels

    def create_loss_tensor(self, predicted_labels, labels):
        """
        Create the loss op and add it to the graph.

        :param predicted_labels: The labels predicted by the graph.
        :type predicted_labels: tf.Tensor
        :param labels: The ground truth labels.
        :type labels: tf.Tensor
        :return: The loss tensor.
        :rtype: tf.Tensor
        """
        return self.relative_differences(predicted_labels, labels)

    @staticmethod
    def relative_differences(predicted_labels, labels):
        """
        Determines the absolute L1 relative differences between two label maps.

        :param predicted_labels: The first label map tensor (usually the predicted labels).
        :type predicted_labels: tf.Tensor
        :param labels: The second label map tensor (usually the actual labels).
        :type labels: tf.Tensor
        :return: The difference tensor.
        :rtype: tf.Tensor
        """
        difference = tf.abs(predicted_labels - labels)
        return difference / labels

    def create_training_op(self, value_to_minimize):
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

    def image_comparison_summary(self, images, labels, predicted_labels, label_differences):
        """
        Combines the image, label, and difference tensors together into a presentable image. Then adds the
        image summary op to the graph.

        :param images: The original image.
        :type images: tf.Tensor
        :param labels: The tensor containing the actual label values.
        :type labels: tf.Tensor
        :param predicted_labels: The tensor containing the predicted labels.
        :type predicted_labels: tf.Tensor
        :param label_differences: The tensor containing the difference between the actual and predicted labels.
        :type label_differences: tf.Tensor
        """
        label_heat_map = self.convert_to_heat_map_rgb(labels)
        predicted_label_heat_map = self.convert_to_heat_map_rgb(predicted_labels)
        label_difference_heat_map = self.convert_to_heat_map_rgb(label_differences)

        comparison_image = tf.concat(1, [images, label_heat_map, predicted_label_heat_map, label_difference_heat_map])
        tf.image_summary('comparison', comparison_image)

    def train_network(self):
        """
        Adds the training operations and runs the training loop.
        """
        print('Preparing data...')
        # Setup the inputs.
        images_tensor, labels_tensor = self.data.inputs(data_type='train', batch_size=self.batch_size,
                                                        num_epochs=self.number_of_epochs)

        print('Building graph...')
        # Add the forward pass operations to the graph.
        self.dropout_keep_probability = tf.placeholder(tf.float32)
        predicted_labels_tensor = self.linear_classifier_inference(images_tensor)

        # Add the loss operations to the graph.
        with tf.name_scope('loss'):
            loss_tensor = self.create_loss_tensor(predicted_labels_tensor, labels_tensor)
            loss_per_pixel_tensor = tf.reduce_mean(loss_tensor)
            tf.scalar_summary("Loss per pixel", loss_per_pixel_tensor)

        with tf.name_scope('comparison_summary'):
            self.image_comparison_summary(images_tensor, labels_tensor, predicted_labels_tensor, loss_tensor)

        # Add the training operations to the graph.
        training_op = self.create_training_op(value_to_minimize=loss_per_pixel_tensor)

        # The op for initializing the variables.
        initialize_op = tf.initialize_all_variables()

        # Prepare the saver.
        saver = tf.train.Saver()

        # Create a session for running operations in the Graph.
        session = tf.Session()

        # Prepare the summary operations.
        summaries_op = tf.merge_all_summaries()
        summary_path = os.path.join(self.log_directory, datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))
        writer = tf.train.SummaryWriter(summary_path, session.graph_def)

        print('Starting training...')
        # Initialize the variables.
        session.run(initialize_op)

        # Start input enqueue threads.
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        # Preform the training loop.
        step = 0
        stop_signal = False
        try:
            while not coordinator.should_stop() and not stop_signal:
                # Regular training step.
                start_time = time.time()
                _, loss, summaries = session.run(
                    [training_op, loss_per_pixel_tensor, summaries_op],
                    feed_dict={self.dropout_keep_probability: 0.5}
                )
                duration = time.time() - start_time

                # Information print and summary write step.
                if step % self.summary_step_period == 0:
                    writer.add_summary(summaries, step)
                    print('Step %d: Loss per pixel = %.5f (%.3f sec / step)' % (step, loss, duration))
                step += 1

                # If a stop has been called for clean up and save.
                if self.queue:
                    if not self.queue.empty():
                        message = self.queue.get(block=False)
                        if message == 'save':
                            save_path = saver.save(session, os.path.join('models', 'depthnet.ckpt'),
                                                   global_step=step)
                            tf.train.write_graph(session.graph_def, 'models', 'depthnet.pb')
                            print("Model saved in file: %s" % save_path)
                        if message == 'quit':
                            stop_signal = True
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (self.number_of_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coordinator.request_stop()

        # Wait for threads to finish.
        coordinator.join(threads)
        session.close()

    def run(self):
        """
        Allow for training the network from a multiprocessing standpoint.
        """
        self.train_network()


if __name__ == '__main__':
    interface = Interface(network_class=DepthNet)
    interface.train()
