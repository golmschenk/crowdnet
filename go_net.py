"""
Code related to the DepthNet.
"""
import datetime
import multiprocessing
import os
import time
import tensorflow as tf
import numpy as np

from convenience import weight_variable, bias_variable, conv2d, leaky_relu, size_from_stride_two
from go_data import GoData
from interface import Interface


class GoNet(multiprocessing.Process):
    """
    The class to build and interact with the GoNet TensorFlow graph.
    """

    def __init__(self, message_queue=None):
        super().__init__()

        # Common variables.
        self.batch_size = 8
        self.number_of_epochs = 50000
        self.initial_learning_rate = 0.00001
        self.data = GoData(data_name='nyud')
        self.summary_step_period = 1
        self.log_directory = 'logs'
        self.dropout_keep_probability = 0.5
        self.network_name = 'go_net'

        # Internal setup.
        self.moving_average_loss = None
        self.moving_average_decay = 0.1
        self.stop_signal = False
        self.step = 0
        self.saver = None
        self.session = None
        self.dropout_keep_probability_tensor = tf.placeholder(tf.float32)
        self.queue = message_queue

    def create_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        return self.create_linear_classifier_inference_op(images)

    def create_linear_classifier_inference_op(self, images):
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
        Determines the L1 relative differences between two label maps.

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

    def interface_handler(self):
        """
        Handle input from the user using the interface.
        """
        if self.queue:
            if not self.queue.empty():
                message = self.queue.get(block=False)
                if message == 'save':
                    save_path = self.saver.save(self.session, os.path.join('models', self.network_name + '.ckpt'),
                                                global_step=self.step)
                    tf.train.write_graph(self.session.graph_def, 'models', self.network_name + '.pb')
                    print("Model saved in file: %s" % save_path)
                if message == 'quit':
                    self.stop_signal = True

    def train(self):
        """
        Adds the training operations and runs the training loop.
        """
        print('Preparing data...')
        # Setup the inputs.
        images_tensor, labels_tensor = self.data.inputs(data_type='', batch_size=self.batch_size,
                                                        num_epochs=self.number_of_epochs)

        print('Building graph...')
        # Add the forward pass operations to the graph.
        predicted_labels_tensor = self.create_inference_op(images_tensor)

        # Add the loss operations to the graph.
        with tf.name_scope('loss'):
            loss_tensor = self.create_loss_tensor(predicted_labels_tensor, labels_tensor)
            loss_per_pixel_tensor = tf.reduce_mean(loss_tensor)
            tf.scalar_summary("Loss per pixel", loss_per_pixel_tensor)
            running_average_loss_per_pixel_tensor = tf.Variable(initial_value=-1.0)
            running_average_loss_per_pixel_op = tf.cond(
                tf.equal(running_average_loss_per_pixel_tensor, -1.0),
                lambda: tf.assign(running_average_loss_per_pixel_tensor, loss_per_pixel_tensor),
                lambda: tf.assign(running_average_loss_per_pixel_tensor,
                                  tf.mul(loss_per_pixel_tensor, self.moving_average_decay) +
                                  tf.mul(running_average_loss_per_pixel_tensor, 1 - self.moving_average_decay))
            )
            with tf.control_dependencies([running_average_loss_per_pixel_op]):
                tf.scalar_summary('Running average loss per pixel', running_average_loss_per_pixel_tensor)

        with tf.name_scope('comparison_summary'):
            self.image_comparison_summary(images_tensor, labels_tensor, predicted_labels_tensor, loss_tensor)

        # Add the training operations to the graph.
        training_op = self.create_training_op(value_to_minimize=loss_per_pixel_tensor)

        # The op for initializing the variables.
        initialize_op = tf.initialize_all_variables()

        # Prepare session.
        self.session = tf.Session()

        # Prepare the summary operations.
        summaries_op = tf.merge_all_summaries()
        summary_path = os.path.join(self.log_directory, datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))
        writer = tf.train.SummaryWriter(summary_path, self.session.graph)

        # Prepare saver.
        self.saver = tf.train.Saver()

        print('Starting training...')
        # Initialize the variables.
        self.session.run(initialize_op)

        # Start input enqueue threads.
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.session, coord=coordinator)

        # Preform the training loop.
        try:
            while not coordinator.should_stop() and not self.stop_signal:
                # Regular training step.
                start_time = time.time()
                _, loss, summaries = self.session.run(
                    [training_op, loss_per_pixel_tensor, summaries_op],
                    feed_dict={self.dropout_keep_probability_tensor: self.dropout_keep_probability}
                )
                duration = time.time() - start_time

                # Information print and summary write step.
                if self.step % self.summary_step_period == 0:
                    writer.add_summary(summaries, self.step)
                    print('Step %d: Loss per pixel = %.5f (%.3f sec / step)' % (self.step, loss, duration))
                self.step += 1

                # Handle interface messages from the user.
                self.interface_handler()
        except tf.errors.OutOfRangeError:
            if self.step == 0:
                print('Training data not found.')
            else:
                print('Done training for %d epochs, %d steps.' % (self.number_of_epochs, self.step))
        finally:
            # When done, ask the threads to stop.
            coordinator.request_stop()

        # Wait for threads to finish.
        coordinator.join(threads)
        self.session.close()

    def run(self):
        """
        Allow for training the network from a multiprocessing standpoint.
        """
        self.train()

    def predict(self, model_file_name='crowd_net'):
        """
        Use a trained model to predict labels for a new set of images.

        :param model_file_name: The trained model's file name.
        :type model_file_name: str
        """
        print('Preparing data...')
        # Setup the inputs.
        images_tensor = tf.placeholder(tf.float32, [None, self.data.height, self.data.width, 3])

        print('Building graph...')
        # Add the forward pass operations to the graph.
        predicted_labels_tensor = self.create_inference_op(images_tensor)

        # The op for initializing the variables.
        initialize_op = tf.initialize_all_variables()

        # Prepare the saver.
        saver = tf.train.Saver()

        # Create a session for running operations in the Graph.
        session = tf.Session()

        print('Running prediction...')
        # Initialize the variables.
        session.run(initialize_op)

        saver.restore(session, os.path.join('models', model_file_name))

        # Preform the training loop.
        images = np.load('test_images.npy')
        images = self.data.shrink_array_with_rebinning(images)
        labels = session.run([predicted_labels_tensor], feed_dict={images_tensor: images.astype(np.float32),
                                                                   self.dropout_keep_probability_tensor: 1.0})
        np.save(os.path.join(self.data.data_directory, 'predicted_labels'), labels)

        session.close()
        print('Done.')


if __name__ == '__main__':
    interface = Interface(network_class=GoNet)
    interface.train()
