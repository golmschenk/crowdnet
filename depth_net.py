"""
Code related to the DepthNet.
"""
import math
import os
import datetime
import tensorflow as tf
import time
import multiprocessing

from data import Data


class DepthNet(multiprocessing.Process):
    """
    The class to build and interact with the DepthNet TensorFlow graph.
    """

    def __init__(self, message_queue=None):
        super().__init__()
        self.batch_size = 8
        self.number_of_epochs = 50000
        self.initial_learning_rate = 0.00001
        self.summary_step_period = 1
        self.log_directory = "logs"
        self.data = Data(data_directory='examples', data_name='nyud_micro')
        self.queue = message_queue
        self.dropout_keep_probability = None

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

            h_conv = self.leaky_relu(conv2d(images, w_conv) + b_conv)

        with tf.name_scope('conv2'):
            w_conv = weight_variable([5, 5, 32, 128])
            b_conv = bias_variable([128])

            h_conv = self.leaky_relu(conv2d(h_conv, w_conv) + b_conv)

        for index in range(9):
            with tf.name_scope('conv' + str(index + 3)):
                w_conv = weight_variable([5, 5, 128, 128])
                b_conv = bias_variable([128])

                h_conv = self.leaky_relu(conv2d(h_conv, w_conv) + b_conv)

        with tf.name_scope('conv12'):
            w_conv = weight_variable([5, 5, 128, 32])
            b_conv = bias_variable([32])

            h_conv = self.leaky_relu(conv2d(h_conv, w_conv) + b_conv)

        with tf.name_scope('fc1'):
            fc0_size = self.data.height * self.data.width * 32
            fc1_size = fc0_size // 4096
            h_fc = tf.reshape(h_conv, [-1, fc0_size])
            w_fc = weight_variable([fc0_size, fc1_size])
            b_fc = bias_variable([fc1_size])

            h_fc = self.leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)
            h_fc_drop = tf.nn.dropout(h_fc, self.dropout_keep_probability)

        with tf.name_scope('fc2'):
            fc2_size = fc1_size // 2
            w_fc = weight_variable([fc1_size, fc2_size])
            b_fc = bias_variable([fc2_size])

            h_fc = self.leaky_relu(tf.matmul(h_fc_drop, w_fc) + b_fc)
            h_fc_drop = tf.nn.dropout(h_fc, self.dropout_keep_probability)

        with tf.name_scope('fc3'):
            fc3_size = self.data.height * self.data.width
            w_fc = weight_variable([fc2_size, fc3_size])
            b_fc = bias_variable([fc3_size])

            h_fc = self.leaky_relu(tf.matmul(h_fc_drop, w_fc) + b_fc)
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

            h_conv = self.leaky_relu(conv2d(images, w_conv) + b_conv)

        with tf.name_scope('conv2'):
            w_conv = weight_variable([7, 7, 16, 24])
            b_conv = bias_variable([24])

            h_conv = self.leaky_relu(conv2d(h_conv, w_conv, [1, 2, 2, 1]) + b_conv)

        with tf.name_scope('conv3'):
            w_conv = weight_variable([7, 7, 24, 32])
            b_conv = bias_variable([32])

            h_conv = self.leaky_relu(conv2d(h_conv, w_conv, [1, 2, 2, 1]) + b_conv)

        with tf.name_scope('fc1'):
            fc0_size = self.size_from_stride_four(self.data.height) * self.size_from_stride_four(self.data.width) * 32
            fc1_size = fc0_size // 2
            h_fc = tf.reshape(h_conv, [-1, fc0_size])
            w_fc = weight_variable([fc0_size, fc1_size])
            b_fc = bias_variable([fc1_size])

            h_fc = self.leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)

        with tf.name_scope('fc2'):
            fc2_size = fc1_size // 2
            w_fc = weight_variable([fc1_size, fc2_size])
            b_fc = bias_variable([fc2_size])

            h_fc = self.leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)

        with tf.name_scope('fc3'):
            fc3_size = self.data.height * self.data.width
            w_fc = weight_variable([fc2_size, fc3_size])
            b_fc = bias_variable([fc3_size])

            h_fc = self.leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)
            predicted_labels = tf.reshape(h_fc, [-1, self.data.height, self.data.width, 1])

        return predicted_labels

    @staticmethod
    def size_from_stride_two(size):
        """
        Provides the appropriate size that will be output with a stride two filter.

        :param size: The original size.
        :type size: int
        :return: The filter output size.
        :rtype: int
        """
        return math.ceil(size / 2)

    def size_from_stride_four(self, size):
        """
        Provides the appropriate size that will be output with a stride four filter.

        :param size: The original size.
        :type size: int
        :return: The filter output size.
        :rtype: int
        """
        return self.size_from_stride_two(self.size_from_stride_two(size))

    def size_from_stride_eight(self, size):
        """
        Provides the appropriate size that will be output with a stride eight filter.

        :param size: The original size.
        :type size: int
        :return: The filter output size.
        :rtype: int
        """
        return self.size_from_stride_four(self.size_from_stride_four(size))

    def linear_classifier_inference(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using only a linear classifier.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        self.initial_learning_rate = 0.00001
        pixel_count = Data().height * Data().width
        flat_images = tf.reshape(images, [-1, pixel_count * Data().channels])
        weights = weight_variable([pixel_count * Data().channels, pixel_count], stddev=0.001)
        biases = bias_variable([pixel_count], constant=0.001)

        flat_predicted_labels = tf.matmul(flat_images, weights) + biases
        predicted_labels = tf.reshape(flat_predicted_labels, [-1, Data().height, Data().width, 1])
        return predicted_labels

    @staticmethod
    def leaky_relu(x):
        """
        A basic implementation of a leaky ReLU.

        :param x: The input of the ReLU activation.
        :type x: tf.Tensor
        :return: The tensor filtering on the leaky activation.
        :rtype: tf.Tensor
        """
        return tf.maximum(tf.mul(0.001, x), x)

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

    def side_by_side_image_summary(self, images, labels, predicted_labels, label_differences):
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
        with tf.Graph().as_default():
            print('Preparing data...')
            # Setup the inputs.
            images, labels = self.data.inputs(data_type='train', batch_size=self.batch_size,
                                              num_epochs=self.number_of_epochs)

            print('Building graph...')
            # Add the forward pass operations to the graph.
            self.dropout_keep_probability = tf.placeholder(tf.float32)
            predicted_labels = self.inference(images)

            # Add the loss operations to the graph.
            with tf.name_scope('loss'):
                relative_differences = self.relative_differences(predicted_labels, labels)
                relative_difference_sum = tf.reduce_sum(relative_differences)
                average_relative_difference = tf.reduce_mean(relative_differences)
                tf.scalar_summary("Loss", relative_difference_sum)
                tf.scalar_summary("Loss per pixel", average_relative_difference)

            with tf.name_scope('comparison_summary'):
                self.side_by_side_image_summary(images, labels, predicted_labels, relative_differences)

            # Add the training operations to the graph.
            train_op = self.training(relative_difference_sum)

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
                    _, average_relative_difference_value, summaries = session.run(
                        [train_op, average_relative_difference, summaries_op],
                        feed_dict={self.dropout_keep_probability: 0.5}
                    )
                    duration = time.time() - start_time

                    # Information print and summary write step.
                    if step % self.summary_step_period == 0:
                        writer.add_summary(summaries, step)
                        print('Step %d: Loss per pixel = %.5f (%.3f sec)' % (step,
                                                                             average_relative_difference_value,
                                                                             duration))
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


def weight_variable(shape, stddev=0.001):
    """
    Create a generic weight variable.

    :param shape: The shape of the weight variable.
    :type shape: list[int]
    :param stddev: The standard deviation to initialize the weights to.
    :type stddev: float
    :return: The weight variable.
    :rtype: tf.Variable
    """
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, constant=0.001):
    """
    Create a generic bias variable.

    :param shape: The shape of the bias variable.
    :type shape: list[int]
    :param constant: The initial value of the biases.
    :type constant: float
    :return: The bias variable.
    :rtype: tf.Variable
    """
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)


def conv2d(images, weights, strides=None):
    """
    Create a generic convolutional operation.

    :param images: The images to prefrom the convolution on.
    :type images: tf.Tensor
    :param weights: The weight variable to be applied.
    :type weights: tf.Variable
    :param strides: The strides to perform the convolution on.
    :type strides: list[int]
    :return: The convolutional operation.
    :rtype: tf.Tensor
    """
    if strides is None:
        strides = [1, 1, 1, 1]
    return tf.nn.conv2d(images, weights, strides=strides, padding='SAME')


if __name__ == '__main__':
    queue_ = multiprocessing.Queue()
    depth_net = DepthNet(message_queue=queue_)
    depth_net.start()
    while True:
        user_input = input()
        if user_input == 's':
            print('Save requested.')
            queue_.put('save')
            continue
        elif user_input == 'q':
            print('Exit requested. Quitting.')
            queue_.put('quit')
            print('Waiting for graph to quit.')
            depth_net.join()
            break
