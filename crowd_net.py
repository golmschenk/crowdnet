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

        reuse = False
        if type == 'validation':
            reuse = True
        with tf.variable_scope('inference', reuse=reuse), tf.name_scope(type):
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
        print('Building graph...')
        train_density_error_tensor, train_count_error_tensor = self.create_network(type='train')
        self.create_network(type='validation')

        loss_tensor = tf.add(train_density_error_tensor, tf.multiply(tf.constant(2.0), train_count_error_tensor))
        training_op = self.create_training_op(loss_tensor)

        checkpoint_directory = os.path.join(self.settings.logs_directory, self.settings.network_name + ' ' +
                                            datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))

        print('Starting training...')
        train_summary_hook = tf.train.SummarySaverHook(save_steps=10,
                                                       output_dir=os.path.join(checkpoint_directory, 'train'),
                                                       summary_op=tf.summary.merge(
                                                           tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                                             scope='train')))
        validation_summary_hook = tf.train.SummarySaverHook(save_steps=10,
                                                            output_dir=os.path.join(checkpoint_directory, 'validation'),
                                                            summary_op=tf.summary.merge(
                                                                tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                                                  scope='validation')))

        checkpoint_saver_hook = tf.train.CheckpointSaverHook(checkpoint_directory, save_steps=100,
                                                             saver=tf.train.Saver())
        with tf.train.MonitoredSession(
                hooks=[train_summary_hook, validation_summary_hook, checkpoint_saver_hook]) as session:
            while True:
                _, loss, step = session.run([training_op, loss_tensor, self.global_step])
                print('Step: {} - Loss: {}'.format(step, loss))

    def old_train(self):
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
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d], outputs_collections=tf.GraphKeys.ACTIVATIONS):
            predicted_labels_tensor = self.create_inference_op(images_tensor)

        # Apply ROI mask.
        negative_one_mask_locations = tf.less_equal(labels_tensor, tf.constant(-1.0))
        labels_tensor = tf.where(negative_one_mask_locations, tf.zeros_like(labels_tensor), labels_tensor)
        predicted_labels_tensor = tf.where(negative_one_mask_locations, tf.zeros_like(predicted_labels_tensor),
                                           predicted_labels_tensor)
        self.predicted_person_count = self.example_mean_pixel_sum(
            tf.where(negative_one_mask_locations, tf.zeros_like(self.predicted_person_count_helper),
                     self.predicted_person_count_helper)
        )

        # Add the loss operations to the graph.
        with tf.variable_scope('loss'):
            loss_tensor = self.create_loss_tensor(predicted_labels_tensor, labels_tensor)
            reduce_mean_loss_tensor = tf.reduce_mean(loss_tensor)
            reduce_sum_loss_tensor = tf.reduce_sum(loss_tensor)
            tf.summary.scalar(self.step_summary_name, reduce_mean_loss_tensor)

        if self.image_summary_on:
            with tf.variable_scope('comparison_summary'):
                self.image_comparison_summary(images_tensor, labels_tensor, predicted_labels_tensor, loss_tensor)

        # Add the training operations to the graph.
        training_op = self.create_training_op(value_to_minimize=reduce_sum_loss_tensor)

        # Gradient and activation summaries
        if self.histograms_on:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
            for variable in variables:
                gradients = tf.gradients(reduce_sum_loss_tensor, variable)
                tf.summary.histogram(variable.name, variable)
                if gradients != [None]:
                    tf.summary.histogram(variable.name + '_gradient', gradients)
            for activation in activations:
                tf.summary.histogram(activation.name, activation)

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
        step = 0
        try:
            while not coordinator.should_stop() and not self.stop_signal:
                # Regular training step.
                start_time = time.time()
                # Summary write step.
                step += 1  # This needs to be replaced.
                if step % self.settings.summary_step_period == 0:
                    _, loss, summaries, step = self.session.run(
                        [training_op, reduce_mean_loss_tensor, summaries_op, self.global_step],
                        feed_dict=self.default_feed_dictionary
                    )
                    train_writer.add_summary(summaries, step)
                else:
                    _, loss, step = self.session.run(
                        [training_op, reduce_mean_loss_tensor, self.global_step],
                        feed_dict=self.default_feed_dictionary
                    )
                duration = time.time() - start_time

                # Information print step.
                if step % self.settings.print_step_period == 0:
                    print('Step %d: %s = %.5f (%.3f sec / step)' % (
                        step, self.step_summary_name, loss, duration))

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
