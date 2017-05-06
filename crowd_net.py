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

        self.histograms_on = False
        self.alternate_loss_on = True
        self.edge_percentage = 0.0

        # Internal variables.
        self.alternate_loss = None
        self.labels_tensor = None
        self.predicted_person_count_helper = None
        self.predicted_person_count = None
        self.predicted_test_labels_average_loss = None
        self.predicted_test_labels_person_count = None
        self.predicted_test_labels_relative_miscount = None
        self.true_labels = None

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.learning_rate_tensor = tf.train.exponential_decay(self.settings.initial_learning_rate,
                                                               self.global_step,
                                                               self.settings.learning_rate_decay_steps,
                                                               self.settings.learning_rate_decay_rate)

    def create_experimental_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a patchwise graph setup.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d],
                                            padding='SAME',
                                            normalizer_fn=None,
                                            activation_fn=leaky_relu,
                                            kernel_size=3):
            module1_output = tf.contrib.layers.conv2d(inputs=images, num_outputs=32)
            module2_output = tf.contrib.layers.conv2d(inputs=module1_output, num_outputs=64)
            module3_output = tf.contrib.layers.conv2d(inputs=module2_output, num_outputs=64)
            module4_output = tf.contrib.layers.conv2d(inputs=module3_output, num_outputs=128)
            module5_output = tf.contrib.layers.conv2d(inputs=module4_output, num_outputs=128)
            module6_output = tf.contrib.layers.conv2d(inputs=module5_output, num_outputs=256)
            module7_output = tf.contrib.layers.conv2d(inputs=module6_output, num_outputs=256)
            with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm], scale=True):
                module8_output = tf.contrib.layers.conv2d(inputs=module7_output, num_outputs=10,
                                                          kernel_size=1)
            module9_output = tf.contrib.layers.conv2d(inputs=module8_output, num_outputs=10,
                                                      kernel_size=1, activation_fn=leaky_relu,
                                                      normalizer_fn=None)
            person_density_output = tf.contrib.layers.conv2d(inputs=module9_output, num_outputs=1,
                                                             kernel_size=1, activation_fn=None,
                                                             normalizer_fn=None)
            person_count_map_output = tf.contrib.layers.conv2d(inputs=module9_output, num_outputs=1,
                                                               kernel_size=1, activation_fn=None,
                                                               normalizer_fn=None)
        return person_density_output, person_count_map_output

    def create_loss_tensors(self, labels_tensor, predicted_labels_tensor, predicted_counts_tensor):

        differences_tensor = predicted_labels_tensor - labels_tensor
        tf.summary.scalar('Mean difference', tf.reduce_mean(differences_tensor))
        absolute_differences_tensor = tf.abs(differences_tensor)
        density_error_tensor = self.example_mean_pixel_sum(absolute_differences_tensor)

        true_person_count_tensor = self.example_mean_pixel_sum(labels_tensor, name='person_count')
        count_error_tensor = tf.abs(true_person_count_tensor - predicted_counts_tensor, name='person_miscount')

        # Create Summaries.
        relative_person_miscount_tensor = tf.divide(count_error_tensor, tf.add(true_person_count_tensor, 0.01),
                                                    name='mean_relative_person_miscount')
        signed_relative_person_miscount_tensor = tf.divide(tf.subtract(predicted_counts_tensor,
                                                                       true_person_count_tensor),
                                                           tf.add(true_person_count_tensor, 0.01),
                                                           name='signed_relative_person_miscount')
        tf.summary.scalar('Density Error', density_error_tensor)
        tf.summary.scalar('Count Error', count_error_tensor)
        tf.summary.scalar('Relative Count Error', relative_person_miscount_tensor)
        tf.summary.scalar('Signed Relative Count Error', signed_relative_person_miscount_tensor)
        tf.summary.scalar('True Count', true_person_count_tensor)
        tf.summary.scalar('Predicted Count', predicted_counts_tensor)

        return density_error_tensor, count_error_tensor

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

    def create_person_count_summaries(self, labels, predicted_labels, predicted_counts):
        """
        Creates the summaries for the counts of people.

        :param labels: The true person density labels.
        :type labels: tf.Tensor
        :param predicted_labels: The predicted person density labels.
        :type predicted_labels: tf.Tensor
        :return: The relative person miscount tensor.
        :rtype: tf.Tensor
        """
        true_person_count_tensor = self.example_mean_pixel_sum(labels, name='person_count')
        predicted_person_count_tensor = self.predicted_person_count
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
    def example_mean_pixel_sum(tensor, name=None):
        """
        Sums the labels per image and takes the mean over the images.

        :param tensor: The person density labels tensor to process.
        :type tensor: tf.Tensor
        :return: The mean count tensor.
        :rtype: tf.Tensor
        """
        example_mean_pixel_sum_tensor = tf.reduce_mean(tf.reduce_sum(tensor, axis=[1, 2, 3]), name=name)
        return example_mean_pixel_sum_tensor

    def image_comparison_summary(self, images, labels, predicted_labels):
        """
        Combines the image, label, and difference tensors together into a presentable image. Then adds the
        image summary op to the graph.

        :param images: The original image.
        :type images: tf.Tensor
        :param labels: The tensor containing the actual label values.
        :type labels: tf.Tensor
        :param predicted_labels: The tensor containing the predicted labels.
        :type predicted_labels: tf.Tensor
        """
        concatenated_labels = tf.concat(axis=1, values=[labels, predicted_labels])
        concatenated_heat_maps = self.convert_to_heat_map_rgb(concatenated_labels)
        display_images = tf.div(images, tf.reduce_max(tf.abs(images)))
        comparison_image = tf.concat(axis=1, values=[display_images, concatenated_heat_maps])
        tf.summary.image('Comparison', comparison_image)

    def create_training_op(self, value_to_minimize):
        """
        Create and add the training op to the graph.

        :param value_to_minimize: The value to train on.
        :type value_to_minimize: tf.Tensor
        :return: The training op.
        :rtype: tf.Operation
        """
        tf.summary.scalar('Learning rate', self.learning_rate_tensor)
        variables_to_train = self.attain_variables_to_train()
        training_op = tf.train.AdamOptimizer(self.learning_rate_tensor).minimize(value_to_minimize,
                                                                                 global_step=self.global_step,
                                                                                 var_list=variables_to_train)
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

    def apply_roi_mask(self, labels_tensor, predicted_labels_tensor, predicted_count_map_tensor):
        negative_one_mask_locations = tf.less_equal(labels_tensor, tf.constant(-1.0))
        masked_labels_tensor = tf.where(negative_one_mask_locations, tf.zeros_like(labels_tensor), labels_tensor)
        masked_predicted_labels_tensor = tf.where(negative_one_mask_locations, tf.zeros_like(predicted_labels_tensor),
                                                  predicted_labels_tensor)
        masked_predicted_count_map_tensor = tf.where(negative_one_mask_locations,
                                                     tf.zeros_like(predicted_count_map_tensor),
                                                     predicted_count_map_tensor)
        return masked_labels_tensor, masked_predicted_labels_tensor, masked_predicted_count_map_tensor

    def create_network(self, type='train'):
        with tf.name_scope('inputs'):
            images_tensor, labels_tensor = self.data.create_input_tensors_for_dataset(
                data_type=type,
                batch_size=self.settings.batch_size
            )

        with tf.variable_scope('inference'), tf.name_scope(type):
            predicted_labels_tensor, predicted_count_maps_tensor = self.create_experimental_inference_op(images_tensor)

            masked_tensors = self.apply_roi_mask(labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor)
            labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor = masked_tensors

            predicted_counts_tensor = self.example_mean_pixel_sum(predicted_count_maps_tensor)

        density_error_tensor, count_error_tensor = self.create_loss_tensors(labels_tensor,
                                                                            predicted_labels_tensor,
                                                                            predicted_counts_tensor)

        if self.image_summary_on:
            self.image_comparison_summary(images_tensor, labels_tensor, predicted_labels_tensor)

        return density_error_tensor, count_error_tensor

    def train(self):
        print('Building train graph...')
        train_density_error_tensor, train_count_error_tensor = self.create_network(type='train')
        loss_tensor = tf.add(train_density_error_tensor, tf.multiply(tf.constant(2.0), train_count_error_tensor))
        training_op = self.create_training_op(loss_tensor)
        checkpoint_directory = os.path.join(self.settings.logs_directory, self.settings.network_name + ' ' +
                                            datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))

        print('Building validation graph...')
        validation_graph = tf.Graph()
        with validation_graph.as_default(), tf.device('/cpu:0'):
            self.create_network(type='validation')
            validation_summaries = tf.summary.merge_all()
            validation_saver = tf.train.Saver()
            validation_summary_writer = tf.summary.FileWriter(checkpoint_directory + '_validation')
            validation_session = tf.train.MonitoredSession()
            latest_validated_checkpoint_path = None

        print('Starting training...')
        with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_directory + '_train', save_checkpoint_secs=100) as session:
            while True:
                _, loss, step = session.run([training_op, loss_tensor, self.global_step])
                print('Step: {} - Loss: {}'.format(step, loss))
                # Run validation if there's a new checkpoint to validate.
                latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_directory + '_train')
                if latest_checkpoint_path != latest_validated_checkpoint_path:
                    print('Running validation summaries...')
                    validation_saver.restore(validation_session, latest_checkpoint_path)
                    validation_summary_writer.add_summary(validation_session.run(validation_summaries),
                                                          global_step=step)
                    latest_validated_checkpoint_path = latest_checkpoint_path


if __name__ == '__main__':
    interface = Interface(network_class=CrowdNet)
    interface.run()
