"""
Code related to the CrowdNet.
"""
import datetime
import tensorflow as tf
import numpy as np
import os

import time
from gonet.net import Net
from gonet.interface import Interface
from gonet.convenience import weight_variable, bias_variable, leaky_relu, conv2d

from crowd_data import CrowdData
from settings import Settings


class CrowdNet(Net):
    """
    A neural network class to estimate crowd density from single 2D images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(settings=Settings(), *args, **kwargs)

        self.data = CrowdData()

        self.alternate_loss_on = False
        self.edge_percentage = 0.0

        # Internal variables.
        self.alternate_loss = None
        self.labels_tensor = None
        self.predicted_test_labels_average_loss = None
        self.predicted_test_labels_person_count = None
        self.predicted_test_labels_relative_miscount = None
        self.true_labels = None

    def create_loss_tensor(self, predicted_labels, labels):
        """
        Create the loss op and add it to the graph.
        Overrides the GoNet method of the same name.

        :param predicted_labels: The labels predicted by the graph.
        :type predicted_labels: tf.Tensor
        :param labels: The ground truth labels.
        :type labels: tf.Tensor
        :return: The loss tensor.
        :rtype: tf.Tensor
        """

        differences = predicted_labels - labels
        tf.summary.scalar('Mean difference', tf.reduce_mean(differences))
        absolute_differences_tensor = tf.abs(differences)
        if self.edge_percentage > 0.001:
            edge_width = int(self.settings.image_width * self.edge_percentage)
            edge_height = int(self.settings.image_height * self.edge_percentage)
            absolute_differences_tensor = absolute_differences_tensor[:, edge_height:-edge_height,
                                          edge_width:-edge_width]
            padding = [[0, 0], [edge_height, edge_height], [edge_width, edge_width], [0, 0]]
            absolute_differences_tensor = tf.pad(absolute_differences_tensor, padding)

        relative_person_miscount_tensor = self.create_person_count_summaries(labels, predicted_labels)
        if self.alternate_loss_on:
            self.alternate_loss = self.session.graph.get_tensor_by_name('loss/person_miscount:0')
        return absolute_differences_tensor

    def create_person_count_summaries(self, labels, predicted_labels):
        """
        Creates the summaries for the counts of people.

        :param labels: The true person density labels.
        :type labels: tf.Tensor
        :param predicted_labels: The predicted person density labels.
        :type predicted_labels: tf.Tensor
        :return: The relative person miscount tensor.
        :rtype: tf.Tensor
        """
        true_person_count_tensor = self.mean_person_count_for_labels(labels, name='mean_person_count')
        predicted_person_count_tensor = self.mean_person_count_for_labels(predicted_labels,
                                                                          name='predicted_mean_person_count')
        person_miscount_tensor = tf.abs(true_person_count_tensor - predicted_person_count_tensor,
                                        name='person_miscount')
        relative_person_miscount_tensor = tf.divide(person_miscount_tensor, tf.add(true_person_count_tensor, 0.01),
                                                    name='mean_relative_person_miscount')
        signed_relative_person_miscount_tensor = tf.divide(tf.subtract(predicted_person_count_tensor,
                                                                       true_person_count_tensor),
                                                           tf.add(true_person_count_tensor, 0.01),
                                                           name='signed_relative_person_miscount')
        tf.summary.scalar('Signed Relative Person Miscount', signed_relative_person_miscount_tensor)
        tf.summary.scalar('True person count', true_person_count_tensor)
        tf.summary.scalar('Predicted person count', predicted_person_count_tensor)
        tf.summary.scalar('Person miscount', person_miscount_tensor)
        tf.summary.scalar('Relative person miscount', relative_person_miscount_tensor)
        return relative_person_miscount_tensor

    @staticmethod
    def mean_person_count_for_labels(labels_tensor, name=None):
        """
        Sums the labels per image and takes the mean over the images.

        :param labels_tensor: The person density labels tensor to process.
        :type labels_tensor: tf.Tensor
        :return: The mean count tensor.
        :rtype: tf.Tensor
        """
        mean_person_count_tensor = tf.reduce_mean(tf.reduce_sum(labels_tensor, axis=[1, 2]), name=name)
        return mean_person_count_tensor

    def create_patchwise_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a patchwise graph setup.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        with tf.name_scope('conv1'):
            w_conv = weight_variable([3, 3, self.settings.image_depth, 32])
            b_conv = bias_variable([32])

            h_conv = leaky_relu(conv2d(images, w_conv) + b_conv)

        with tf.name_scope('conv2'):
            w_conv = weight_variable([3, 3, 32, 64])
            b_conv = bias_variable([64])

            h_conv = leaky_relu(conv2d(h_conv, w_conv) + b_conv)

        with tf.name_scope('conv3'):
            w_conv = weight_variable([3, 3, 64, 64])
            b_conv = bias_variable([64])

            h_conv = leaky_relu(conv2d(h_conv, w_conv) + b_conv)

        with tf.name_scope('conv4'):
            w_conv = weight_variable([3, 3, 64, 128])
            b_conv = bias_variable([128])

            h_conv = leaky_relu(conv2d(h_conv, w_conv) + b_conv)
            h_conv = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)

        with tf.name_scope('conv5'):
            w_conv = weight_variable([3, 3, 128, 128])
            b_conv = bias_variable([128])

            h_conv = leaky_relu(conv2d(h_conv, w_conv) + b_conv)
            h_conv = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)

        with tf.name_scope('conv6'):
            w_conv = weight_variable([3, 3, 128, 256])
            b_conv = bias_variable([256])

            h_conv = leaky_relu(conv2d(h_conv, w_conv) + b_conv)
            h_conv = tf.nn.dropout(h_conv, self.dropout_keep_probability_tensor)

        with tf.name_scope('conv7'):
            w_conv = weight_variable([7, 7, 256, 1])
            b_conv = bias_variable([1])

            h_conv = conv2d(h_conv, w_conv) + b_conv

        predicted_labels = h_conv
        return predicted_labels

    def create_gaea_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a patchwise graph setup.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """

        module1_output = self.terra_module('module1', images, 32)
        module2_output = self.terra_module('module2', module1_output, 64)
        module3_output = self.terra_module('module3', module2_output, 64)
        module4_output = self.terra_module('module4', module3_output, 128)
        module5_output = self.terra_module('module5', module4_output, 128)
        module6_output = self.terra_module('module6', module5_output, 256)
        module7_output = self.terra_module('module7', module6_output, 256, kernel_size=7, dropout_on=True)
        module8_output = self.terra_module('module8', module7_output, 10, kernel_size=1, dropout_on=True)
        module9_output = self.terra_module('module9', module8_output, 1, kernel_size=1, activation_function=None)
        predicted_labels = module9_output
        return predicted_labels

    def create_gaea_with_final_tanh_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a patchwise graph setup.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """

        module1_output = self.terra_module('module1', images, 32)
        module2_output = self.terra_module('module2', module1_output, 64)
        module3_output = self.terra_module('module3', module2_output, 64)
        module4_output = self.terra_module('module4', module3_output, 128)
        module5_output = self.terra_module('module5', module4_output, 128)
        module6_output = self.terra_module('module6', module5_output, 256)
        module7_output = self.terra_module('module7', module6_output, 256, kernel_size=7, dropout_on=True)
        module8_output = self.terra_module('module8', module7_output, 10, kernel_size=1, dropout_on=True,
                                           activation_function=tf.nn.tanh)
        module9_output = self.terra_module('module9', module8_output, 1, kernel_size=1, activation_function=None)
        predicted_labels = module9_output
        return predicted_labels

    def create_gaea_with_depth_split_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a patchwise graph setup.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        depth = tf.expand_dims(images[:, :, :, 3], axis=3)
        depth_module1_output = self.terra_module('depth_module1', depth, 4)
        depth_module2_output = self.terra_module('depth_module2', depth_module1_output, 8)
        depth_module3_output = self.terra_module('depth_module3', depth_module2_output, 16)
        depth_module4_output = self.terra_module('depth_module4', depth_module3_output, 32)
        depth_module5_attention = self.terra_module('depth_module5', depth_module4_output, 128,
                                                    activation_function=tf.nn.tanh)

        module1_output = self.terra_module('module1', images[:, :, :, :3], 32)
        module2_output = self.terra_module('module2', module1_output, 64)
        module3_output = self.terra_module('module3', module2_output, 64)
        module4_output = self.terra_module('module4', module3_output, 128)
        module5_output = self.terra_module('module5', module4_output, 128)
        module5_output_depth_attention = tf.multiply(module5_output, depth_module5_attention)
        module6_output = self.terra_module('module6', module5_output_depth_attention, 256)
        module7_output = self.terra_module('module7', module6_output, 256, kernel_size=7, dropout_on=True)
        module8_output = self.terra_module('module8', module7_output, 10, kernel_size=1, dropout_on=True)
        module9_output = self.terra_module('module9', module8_output, 1, kernel_size=1, activation_function=None)
        predicted_labels = module9_output
        return predicted_labels

    def create_gaea_with_depth_skip_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a patchwise graph setup.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        depth = tf.expand_dims(images[:, :, :, 3], axis=3)
        module1_output = self.terra_module('module1', images, 32)
        module2_output = self.terra_module('module2', module1_output, 64)
        module3_output = self.terra_module('module3', module2_output, 64)
        module3_output_with_depth = tf.concat([module3_output, depth], axis=3)
        module4_output = self.terra_module('module4', module3_output_with_depth, 128)
        module5_output = self.terra_module('module5', module4_output, 128)
        module6_output = self.terra_module('module6', module5_output, 256)
        module6_output_with_depth = tf.concat([module6_output, depth], axis=3)
        module7_output = self.terra_module('module7', module6_output_with_depth, 256, kernel_size=7, dropout_on=True)
        module8_output = self.terra_module('module8', module7_output, 10, kernel_size=1, dropout_on=True)
        module9_output = self.terra_module('module9', module8_output, 1, kernel_size=1, activation_function=None)
        predicted_labels = module9_output
        return predicted_labels

    def image_comparison_summary(self, images, labels, predicted_labels, label_differences):
        """
        Combines the image, label, and difference tensors together into a presentable image. Then adds the
        image summary op to the graph. Handles images that include depth maps as well.

        :param images: The original image.
        :type images: tf.Tensor
        :param labels: The tensor containing the actual label values.
        :type labels: tf.Tensor
        :param predicted_labels: The tensor containing the predicted labels.
        :type predicted_labels: tf.Tensor
        :param label_differences: The tensor containing the difference between the actual and predicted labels.
        :type label_differences: tf.Tensor
        """
        if self.settings.image_depth == 4:
            concatenated_labels = tf.concat(axis=1, values=[labels, predicted_labels, label_differences])
            concatenated_heat_maps = self.convert_to_heat_map_rgb(concatenated_labels)
            display_images = tf.divide(images[:, :, :, :3], tf.reduce_max(tf.abs(images[:, :, :, :3])))
            depth_image = tf.expand_dims(images[:, :, :, 3], -1)
            depth_heat_map = self.convert_to_heat_map_rgb(depth_image)
            comparison_image = tf.concat(axis=1, values=[display_images, concatenated_heat_maps, depth_heat_map])
            tf.summary.image('comparison', comparison_image)
        else:
            super().image_comparison_summary(images, labels, predicted_labels, label_differences)

    def attain_variables_to_train(self):
        """
        Gets the list of variables to train based on the scopes to train list.

        :return: The list of variables to train.
        :rtype: list[tf.Variable]
        """
        if self.settings.scopes_to_train:
            return [variable for scope in self.settings.scopes_to_train for variable in
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator/'+scope)]
        else:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    def create_training_op(self, value_to_minimize):
        """
        Create and add the training op to the graph.

        :param value_to_minimize: The value to train on.
        :type value_to_minimize: tf.Tensor
        :return: The training op.
        :rtype: tf.Operation
        """
        variables_to_train = self.attain_variables_to_train()
        tf.summary.scalar('Learning rate', self.learning_rate_tensor)
        training_op = tf.train.AdamOptimizer(self.learning_rate_tensor).minimize(
            value_to_minimize,
            global_step=self.global_step,
            var_list=variables_to_train
        )
        if self.alternate_loss_on:
            alternate_training_op = tf.train.AdamOptimizer(self.learning_rate_tensor).minimize(
                self.alternate_loss,
                global_step=None,  # Don't increment the global step on each optimizer.
                var_list=variables_to_train
            )
            training_op = tf.group(training_op, alternate_training_op)
        return training_op

    def create_input_tensors(self):
        """
        Create the image and label tensors for each dataset and produces a selector tensor to choose between datasets
        during runtime.

        :return: The general images and labels tensors which are conditional on a selector tensor.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        images_tensor, labels_tensor = super().create_input_tensors()
        self.labels_tensor = labels_tensor
        return images_tensor, labels_tensor

    def test_run_preloop(self):
        """
        The code run before the test loop. Mostly for setting up things that will be used within the loop.
        """
        with tf.variable_scope('loss'):
            loss_tensor = self.create_loss_tensor(
                self.session.graph.get_tensor_by_name('inference_op:0'),
                self.session.graph.get_tensor_by_name('labels_input_op:0')
            )
            reduce_mean_loss_tensor = tf.reduce_mean(loss_tensor, name='mean_loss_per_pixel')
        self.predicted_test_labels = np.ndarray(shape=[0] + list(self.settings.label_shape), dtype=np.float32)
        self.true_labels = np.ndarray(shape=[0] + list(self.settings.label_shape), dtype=np.float32)
        self.predicted_test_labels_average_loss = np.empty(shape=[], dtype=np.float32)
        self.predicted_test_labels_person_count = np.empty(shape=[], dtype=np.float32)
        self.predicted_test_labels_relative_miscount = np.empty(shape=[], dtype=np.float32)

    def test_run_loop_step(self):
        """
        The code that will be used during the each iteration of the test loop (excluding the step incrementation).
        """
        predicted_labels_tensor = self.session.graph.get_tensor_by_name('inference_op:0')
        predicted_labels_average_loss_tensor = self.session.graph.get_tensor_by_name('loss/mean_loss_per_pixel:0')
        predicted_labels_person_count_tensor = self.session.graph.get_tensor_by_name(
            'loss/predicted_mean_person_count:0')
        true_labels_tensor = self.session.graph.get_tensor_by_name('labels_input_op:0')
        predicted_labels_relative_miscount_tensor = self.session.graph.get_tensor_by_name(
            'loss/mean_relative_person_miscount:0')
        run_result = self.session.run([predicted_labels_tensor, predicted_labels_average_loss_tensor,
                                       predicted_labels_person_count_tensor, predicted_labels_relative_miscount_tensor,
                                       true_labels_tensor],
                                      feed_dict={**self.default_feed_dictionary,
                                                 self.dropout_keep_probability_tensor: 1.0})
        predicted_labels_batch, predicted_labels_average_loss_batch = run_result[0], run_result[1]
        predicted_labels_person_count_batch, predicted_labels_relative_miscount_batch = run_result[2], run_result[3]
        true_labels_batch = run_result[4]
        self.predicted_test_labels = np.concatenate((self.predicted_test_labels, predicted_labels_batch))
        self.true_labels = np.concatenate((self.true_labels, true_labels_batch))
        self.predicted_test_labels_average_loss = np.append(self.predicted_test_labels_average_loss,
                                                            predicted_labels_average_loss_batch)
        self.predicted_test_labels_person_count = np.append(self.predicted_test_labels_person_count,
                                                            predicted_labels_person_count_batch)
        self.predicted_test_labels_relative_miscount = np.append(self.predicted_test_labels,
                                                                 predicted_labels_relative_miscount_batch)
        print('{image_count} images processed.'.format(image_count=(self.test_step + 1) * self.settings.batch_size))

    @staticmethod
    def reset_graph():
        """
        Reset the TensorFlow graph.
        """
        tf.reset_default_graph()

    def generator(self):
        noise = tf.random_uniform([self.settings.batch_size, 1, 1, 50])
        net = tf.contrib.layers.conv2d_transpose(noise, 1024, kernel_size=[4, 4], stride=[1, 1], padding='VALID')
        net = tf.contrib.layers.conv2d_transpose(net, 512, kernel_size=[5, 5], stride=[3, 3], padding='SAME')
        net = tf.contrib.layers.conv2d_transpose(net, 256, kernel_size=[5, 5], stride=[2, 2], padding='SAME')
        net = tf.contrib.layers.conv2d_transpose(net, 128, kernel_size=[5, 5], stride=[2, 2], padding='SAME')
        net = tf.contrib.layers.conv2d_transpose(net, 3, kernel_size=[5, 5], stride=[2, 2], padding='SAME',
                                                 activation_fn=None)
        unscaled_images = net[:, :self.settings.image_height, :self.settings.image_width, :]
        mean, variance = tf.nn.moments(unscaled_images, axes=[1, 2, 3], keep_dims=True)
        images = (unscaled_images - mean) / tf.sqrt(variance)
        return images

    def train(self):
        """
        Adds the training operations and runs the training loop.
        """
        # Prepare session.
        self.session = tf.Session()

        print('Preparing data...')
        # Setup the inputs.
        with tf.variable_scope('Input'):
            images_tensor, labels_tensor = self.create_input_tensors()

        print('Building graph...')
        # Add the forward pass operations to the graph.
        with tf.variable_scope('Generator'):
            generated_images_tensor = self.generator()
            tf.summary.image('Generated Images', generated_images_tensor)
        with tf.variable_scope('Discriminator') as scope:
            predicted_labels_tensor = self.create_inference_op(images_tensor)
            scope.reuse_variables()
            predicted_generated_labels_tensor = self.create_inference_op(generated_images_tensor)

        # Add the loss operations to the graph.
        with tf.variable_scope('loss'):
            loss_tensor = self.create_loss_tensor(predicted_labels_tensor, labels_tensor)
            reduce_mean_loss_tensor = tf.reduce_mean(loss_tensor)
            tf.summary.scalar(self.step_summary_name, reduce_mean_loss_tensor)
            generated_loss_tensor = tf.reduce_mean(tf.abs(predicted_generated_labels_tensor))
            tf.summary.scalar('Generated Loss', generated_loss_tensor)

        if self.image_summary_on:
            with tf.variable_scope('comparison_summary'):
                self.image_comparison_summary(images_tensor, labels_tensor, predicted_labels_tensor, loss_tensor)

        # Add the training operations to the graph.
        normal_training_op = self.create_training_op(value_to_minimize=reduce_mean_loss_tensor)
        generated_discriminator_training_op = tf.train.AdamOptimizer(self.learning_rate_tensor).minimize(
            generated_loss_tensor,
            global_step=None,  # Only increment during main training op.
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        )
        generator_training_op = tf.train.AdamOptimizer(self.learning_rate_tensor).minimize(
            tf.negative(generated_loss_tensor),
            global_step=None,  # Only increment during main training op.
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        )
        training_op = tf.group(normal_training_op, generated_discriminator_training_op, generator_training_op)

        # Prepare the summary operations.
        summaries_op = tf.summary.merge_all()
        summary_path = os.path.join(self.settings.logs_directory, self.settings.network_name + ' ' +
                                    datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))
        self.log_source_files(summary_path + '_source')
        train_writer = tf.summary.FileWriter(summary_path + '_train', self.session.graph)
        validation_writer = tf.summary.FileWriter(summary_path + '_validation', self.session.graph)

        # The op for initializing the variables.
        initialize_op = tf.global_variables_initializer()

        # Prepare saver.
        self.saver = tf.train.Saver(max_to_keep=self.settings.number_of_models_to_keep)

        print('Initializing graph...')
        # Initialize the variables.
        self.session.run(initialize_op)

        # Restore from saved model if passed.
        if self.settings.restore_model_file_path:
            self.model_restore()

        # Start input enqueue threads.
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.session, coord=coordinator)

        print('Starting training...')
        # Preform the training loop.
        try:
            while not coordinator.should_stop() and not self.stop_signal:
                # Regular training step.
                start_time = time.time()
                _, loss, summaries, step = self.session.run(
                    [training_op, reduce_mean_loss_tensor, summaries_op, self.global_step],
                    feed_dict=self.default_feed_dictionary
                )
                duration = time.time() - start_time

                # Information print step.
                if step % self.settings.print_step_period == 0:
                    print('Step %d: %s = %.5f (%.3f sec / step)' % (
                        step, self.step_summary_name, loss, duration))

                # Summary write step.
                if step % self.settings.summary_step_period == 0:
                    train_writer.add_summary(summaries, step)

                # Validation step.
                if step % self.settings.validation_step_period == 0:
                    start_time = time.time()
                    loss, summaries = self.session.run(
                        [reduce_mean_loss_tensor, summaries_op],
                        feed_dict={**self.default_feed_dictionary,
                                   self.dropout_keep_probability_tensor: 1.0,
                                   self.dataset_selector_tensor: 'validation'}
                    )
                    duration = time.time() - start_time
                    validation_writer.add_summary(summaries, step)
                    print('Validation step %d: %s = %.5g (%.3f sec / step)' % (step, self.step_summary_name,
                                                                               loss, duration))

                if step % self.settings.model_auto_save_step_period == 0 and step != 0:
                    self.save_model()

                # Handle interface messages from the user.
                self.interface_handler()
        except tf.errors.OutOfRangeError as error:
            if self.global_step == 0:
                print('Data not found.')
            else:
                raise error
        finally:
            # When done, ask the threads to stop.
            coordinator.request_stop()

        # Wait for threads to finish.
        coordinator.join(threads)
        self.session.close()


if __name__ == '__main__':
    interface = Interface(network_class=CrowdNet)
    interface.run()
