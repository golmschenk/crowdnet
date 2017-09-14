"""
Code related to the CrowdNet.
"""
import datetime
import tensorflow as tf
import os
import numpy as np
import time

from gonet.net import Net
from gonet.convenience import leaky_relu

from crowd_data import CrowdData
from settings import Settings


class CrowdNet(Net):
    """
    A neural network class to estimate crowd density from single 2D images.
    """

    def __init__(self, settings=None, *args, **kwargs):
        if not settings:
            settings = Settings()
        super().__init__(settings=settings, *args, **kwargs)

        self.feature_matching_parameter = 10000.0
        self.density_to_count_loss_ratio = 10.0
        self.data = CrowdData(settings)

        self.clip_value = 1.0
        self.histograms_on = False
        self.alternate_loss_on = True
        self.edge_percentage = 0.0
        self.generator_train_step_period = 5
        self.border_size = 5

        # Internal variables.
        self.lookup_dictionary = {}
        self.alternate_loss = None
        self.labels_tensor = None
        self.predicted_person_count_helper = None
        self.predicted_person_count = None
        self.predicted_test_labels_average_loss = None
        self.predicted_test_labels_person_count = None
        self.predicted_test_labels_relative_miscount = None
        self.true_labels = None
        self.current_data = None
        self.middle_layer_outputs = {}

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.learning_rate_tensor = tf.train.exponential_decay(self.settings.initial_learning_rate,
                                                               self.global_step,
                                                               self.settings.learning_rate_decay_steps,
                                                               self.settings.learning_rate_decay_rate)

        self.average_train_count = self.data.get_average_person_count()

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
            module1_output = tf.contrib.layers.conv2d(inputs=images, num_outputs=16, normalizer_fn=None)
            module2_output = tf.contrib.layers.conv2d(inputs=module1_output, num_outputs=32)
            module3_output = tf.contrib.layers.conv2d(inputs=module2_output, num_outputs=64)
            module4_output = tf.contrib.layers.conv2d(inputs=module3_output, num_outputs=128)
            if self.current_data:
                self.middle_layer_outputs[self.current_data] = module4_output
            module5_output = tf.contrib.layers.conv2d(inputs=module4_output, num_outputs=10, kernel_size=1)
            person_density_output = tf.contrib.layers.conv2d(inputs=module5_output, num_outputs=1, kernel_size=1,
                                                             activation_fn=None, normalizer_fn=None)
            person_count_map_output = tf.contrib.layers.conv2d(inputs=module5_output, num_outputs=1, kernel_size=1,
                                                               activation_fn=None, normalizer_fn=None)
        return person_density_output, person_count_map_output

    def create_error_tensors(self, labels_tensor, predicted_labels_tensor, predicted_counts_tensor):
        """
        Creates the error tensors from the crowd prediction results.
        
        :param labels_tensor: The true density labels.
        :type labels_tensor: tf.Tensor
        :param predicted_labels_tensor: The predicted density labels.
        :type predicted_labels_tensor: tf.Tensor
        :param predicted_counts_tensor: The total predicted counts for an image (may be different than sum of predicted 
                                        density).
        :type predicted_counts_tensor: tf.Tensor
        :return: The error in the densities and the error in the person counts. 
        :rtype: (tf.Tensor, tf.Tensor)
        """
        differences_tensor = tf.subtract(predicted_labels_tensor, labels_tensor)
        tf.summary.scalar('Mean difference', tf.reduce_mean(differences_tensor))
        absolute_differences_tensor = tf.abs(differences_tensor)
        density_error_tensor = self.example_mean_pixel_sum(absolute_differences_tensor)

        per_example_true_person_count_tensor = tf.reduce_sum(labels_tensor, axis=[1, 2, 3])
        count_error_tensor = tf.reduce_mean(tf.abs(tf.subtract(per_example_true_person_count_tensor, predicted_counts_tensor)))

        # Create Summaries.
        relative_person_miscount_tensor = tf.reduce_mean(tf.divide(count_error_tensor, tf.add(per_example_true_person_count_tensor, 0.01)),
                                                         name='mean_relative_person_miscount')
        signed_relative_person_miscount_tensor = tf.reduce_mean(tf.divide(tf.subtract(predicted_counts_tensor,
                                                                          per_example_true_person_count_tensor),
                                                                tf.add(per_example_true_person_count_tensor, 0.01),
                                                                name='signed_relative_person_miscount'))
        tf.summary.scalar('Density Error', density_error_tensor)
        tf.summary.scalar('Count Error', count_error_tensor)
        tf.summary.scalar('Relative Count Error', relative_person_miscount_tensor)
        tf.summary.scalar('Signed Relative Count Error', signed_relative_person_miscount_tensor)
        tf.summary.scalar('True Count', tf.reduce_mean(per_example_true_person_count_tensor))
        tf.summary.scalar('Predicted Count', tf.reduce_mean(predicted_counts_tensor))

        return density_error_tensor, count_error_tensor

    @staticmethod
    def example_mean_pixel_sum(tensor):
        """
        Sums the labels per image and takes the mean over the images.
        
        :param tensor: The person density labels tensor to process.
        :type tensor: tf.Tensor
        :return: The mean count tensor.
        :rtype: tf.Tensor
        """
        example_mean_pixel_sum_tensor = tf.reduce_mean(tf.reduce_sum(tensor, axis=[1, 2, 3]))
        return example_mean_pixel_sum_tensor

    def density_comparison_summary(self, images, labels, predicted_labels):
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
        tf.summary.scalar('Loss', value_to_minimize)
        variables_to_train = self.attain_variables_to_train()
        training_op = tf.train.RMSPropOptimizer(self.learning_rate_tensor).minimize(value_to_minimize,
                                                                                    global_step=self.global_step,
                                                                                    var_list=variables_to_train)
        return training_op

    @staticmethod
    def apply_roi_mask(labels_tensor, predicted_labels_tensor, predicted_count_map_tensor):
        """
        Applies the ROI mask to the sets of labels.
        
        :param labels_tensor: The labels (which include the -1 mask).
        :type labels_tensor: tf.Tensor
        :param predicted_labels_tensor: The predicted density labels.
        :type predicted_labels_tensor: tf.Tensor
        :param predicted_count_map_tensor: The predicted count map.
        :type predicted_count_map_tensor: tf.Tensor
        :return: The update set of tensors with the masked portions zeroed.
        :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
        """
        negative_one_mask_locations = tf.less_equal(labels_tensor, tf.constant(-1.0))
        masked_labels_tensor = tf.where(negative_one_mask_locations, tf.zeros_like(labels_tensor), labels_tensor)
        masked_predicted_labels_tensor = tf.where(negative_one_mask_locations, tf.zeros_like(predicted_labels_tensor),
                                                  predicted_labels_tensor)
        masked_predicted_count_map_tensor = tf.where(negative_one_mask_locations,
                                                     tf.zeros_like(predicted_count_map_tensor),
                                                     predicted_count_map_tensor)
        return masked_labels_tensor, masked_predicted_labels_tensor, masked_predicted_count_map_tensor

    def create_network(self, run_type='train'):
        """
        Create the pieces of the network from input to error.
        
        :param run_type: The running type of this network instance (training, validation, etc).
        :type run_type: string
        :return: The density errors and the person count errors. 
        :rtype: (tf.Tensor, tf.Tensor)
        """
        with tf.name_scope('inputs'):
            images_tensor, labels_tensor, guesses_tensor = self.data.create_input_tensors_for_dataset(
                data_type=run_type,
                batch_size=self.settings.batch_size
            )
            self.lookup_dictionary['labeled_images_tensor'] = images_tensor

        if run_type == 'train':
            dropout_keep_probability = 0.5
        else:
            dropout_keep_probability = 1.0
        dropout_arg_scope = tf.contrib.framework.arg_scope([tf.contrib.layers.dropout],
                                                           keep_prob=dropout_keep_probability)

        with tf.variable_scope('inference'), dropout_arg_scope:
            predicted_labels_tensor, predicted_count_maps_tensor = self.create_experimental_inference_op(images_tensor)

            masked_tensors = self.apply_roi_mask(labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor)
            labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor = masked_tensors
            self.lookup_dictionary['true_predicted_labels'] = predicted_labels_tensor
            self.lookup_dictionary['true_predicted_count_maps'] = predicted_count_maps_tensor

            predicted_counts_tensor = tf.reduce_sum(predicted_count_maps_tensor, axis=[1, 2, 3])

        density_error_tensor, count_error_tensor = self.create_error_tensors(labels_tensor,
                                                                             predicted_labels_tensor,
                                                                             predicted_counts_tensor)

        if self.settings.run_mode == 'test':
            predictor_scope = tf.variable_scope('True/predictor')
        else:
            predictor_scope = tf.variable_scope('predictor')
        with predictor_scope:
            positive_predicted_labels_tensor = tf.maximum(predicted_labels_tensor, 0)
            positive_predicted_count_maps_tensor = tf.maximum(predicted_count_maps_tensor, 0)
            labels_multiplier = tf.Variable(initial_value=tf.constant(1, dtype=positive_predicted_labels_tensor.dtype, shape=[]))
            predictor_predicted_labels_tensor = tf.multiply(positive_predicted_labels_tensor, labels_multiplier)
            counts_multiplier = tf.Variable(initial_value=tf.constant(1, dtype=positive_predicted_count_maps_tensor.dtype, shape=[]))
            predictor_predicted_count_maps_tensor = tf.multiply(positive_predicted_count_maps_tensor, counts_multiplier)
            self.lookup_dictionary['labels_multiplier'] = labels_multiplier
            self.lookup_dictionary['counts_multiplier'] = counts_multiplier
            predictor_predicted_counts_tensor = tf.reduce_sum(predictor_predicted_count_maps_tensor, axis=[1, 2, 3])
        with tf.name_scope('Predictor'):
            predictor_density_error_tensor, predictor_count_error_tensor = self.create_error_tensors(
                labels_tensor,
                predictor_predicted_labels_tensor,
                predictor_predicted_counts_tensor
            )

        self.density_comparison_summary(images_tensor, labels_tensor, predicted_labels_tensor)

        self.lookup_dictionary['labels_tensor'] = labels_tensor
        if self.settings.test_with_predictor:
            self.lookup_dictionary['predicted_labels_tensor'] = predictor_predicted_labels_tensor
            self.lookup_dictionary['predicted_counts_tensor'] = predictor_predicted_counts_tensor
        else:
            self.lookup_dictionary['predicted_labels_tensor'] = predicted_labels_tensor
            self.lookup_dictionary['predicted_counts_tensor'] = predicted_counts_tensor
        self.lookup_dictionary['predictor_density_error_tensor'] = predictor_density_error_tensor
        self.lookup_dictionary['predictor_count_error_tensor'] = predictor_count_error_tensor

        return density_error_tensor, count_error_tensor

    def get_checkpoint_directory_basename(self):
        if self.settings.restore_checkpoint_directory:
            self.settings.restore_checkpoint_directory = self.settings.restore_checkpoint_directory.replace('_train',
                                                                                                            '')
        if self.settings.restore_mode == 'transfer':
            self.settings.restore_checkpoint_directory = self.settings.restore_checkpoint_directory + '_train'
            self.settings.restore_checkpoint_directory = os.path.join(self.settings.logs_directory,
                                                                      self.settings.restore_checkpoint_directory)
            return os.path.join(self.settings.logs_directory, self.settings.network_name + ' ' +
                                datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))
        if self.settings.run_mode == 'test':
            return os.path.join(self.settings.logs_directory, self.settings.restore_checkpoint_directory + '_train')
        elif self.settings.restore_checkpoint_directory and self.settings.restore_mode == 'continue':
            return os.path.join(self.settings.logs_directory, self.settings.restore_checkpoint_directory)
        else:
            return os.path.join(self.settings.logs_directory, self.settings.network_name + ' ' +
                                datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))

    def unlabeled_generator(self):
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.conv2d_transpose],
                                            padding='SAME',
                                            normalizer_fn=tf.contrib.layers.batch_norm,
                                            activation_fn=leaky_relu,
                                            kernel_size=3):
            noise = tf.random_normal([self.settings.batch_size, self.settings.image_height + (self.border_size * 2),
                                      self.settings.image_width + (self.border_size * 2),
                                      50])
            net = tf.contrib.layers.conv2d_transpose(noise, 256, normalizer_fn=None)
            net = tf.contrib.layers.conv2d_transpose(net, 128)
            net = tf.contrib.layers.conv2d_transpose(net, 64)
            net = tf.contrib.layers.conv2d_transpose(net, 32)
            net = tf.contrib.layers.conv2d_transpose(net, 3, activation_fn=tf.tanh, normalizer_fn=None)
            mean, variance = tf.nn.moments(net, axes=[1, 2, 3], keep_dims=True)
            images_tensor = (net - mean) / tf.sqrt(variance)
        return images_tensor[:, self.border_size:-self.border_size, self.border_size:-self.border_size, :]

    def strided_unlabeled_generator(self):
        assert self.settings.image_height is 72 and self.settings.image_width is 90
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.conv2d_transpose],
                                            padding='SAME',
                                            normalizer_fn=tf.contrib.layers.batch_norm,
                                            activation_fn=leaky_relu,
                                            kernel_size=5):
            noise = tf.random_normal([self.settings.batch_size, 1, 1, 50])
            net = tf.contrib.layers.conv2d_transpose(noise, 1024, kernel_size=[4, 5], stride=1, padding='VALID',
                                                     normalizer_fn=None)
            net = tf.contrib.layers.conv2d_transpose(net, 512, stride=3)
            net = tf.contrib.layers.conv2d_transpose(net, 256, stride=3)
            net = tf.contrib.layers.conv2d_transpose(net, 3, stride=2, activation_fn=tf.tanh, normalizer_fn=None)
            mean, variance = tf.nn.moments(net, axes=[1, 2, 3], keep_dims=True)
            images_tensor = (net - mean) / tf.sqrt(variance)
        return images_tensor

    def create_generated_network(self):
        with tf.variable_scope('generator'):
            images_tensor = self.unlabeled_generator()
            tf.summary.image('Generated Images', images_tensor)
            self.lookup_dictionary['generated_images_tensor'] = images_tensor
        dropout_arg_scope = tf.contrib.framework.arg_scope([tf.contrib.layers.dropout],
                                                           keep_prob=0.5)
        with tf.variable_scope('inference', reuse=True), dropout_arg_scope:
            predicted_labels_tensor, predicted_count_maps_tensor = self.create_experimental_inference_op(images_tensor)

        return predicted_labels_tensor, predicted_count_maps_tensor

    def create_unlabeled_inference_network(self):
        with tf.name_scope('unlabeled_inputs'):
            images_tensor, labels_tensor, guesses_tensor = self.data.create_input_tensors_for_dataset(
                data_type='unlabeled',
                batch_size=self.settings.batch_size
            )
            self.lookup_dictionary['unlabeled_images_tensor'] = images_tensor
            self.lookup_dictionary['guesses'] = guesses_tensor
        dropout_arg_scope = tf.contrib.framework.arg_scope([tf.contrib.layers.dropout],
                                                           keep_prob=0.5)
        with tf.variable_scope('inference', reuse=True), dropout_arg_scope:
            predicted_labels_tensor, predicted_count_maps_tensor = self.create_experimental_inference_op(images_tensor)
            self.lookup_dictionary['unlabeled_predicted_labels'] = predicted_labels_tensor
            self.lookup_dictionary['unlabeled_predicted_count_maps'] = predicted_count_maps_tensor

        masked_tensors = self.apply_roi_mask(labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor)
        labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor = masked_tensors

        return predicted_labels_tensor, predicted_count_maps_tensor

    def train(self):
        """
        Runs the training of the network.
        """
        print('Building train graph...')
        train_density_error_tensor, train_count_error_tensor = self.create_network(run_type='train')
        loss_tensor = tf.add(tf.multiply(tf.constant(self.density_to_count_loss_ratio), train_density_error_tensor),
                             train_count_error_tensor)
        training_op = self.create_training_op(loss_tensor)
        checkpoint_directory_basename = self.get_checkpoint_directory_basename()
        if self.settings.restore_checkpoint_directory:
            restorer = tf.train.Saver()
        else:
            restorer = None

        print('Building validation graph...')
        validation_graph = tf.Graph()
        with validation_graph.as_default(), tf.device('/cpu:0'):
            self.create_network(run_type='validation')
            validation_summaries = tf.summary.merge_all()
            validation_saver = tf.train.Saver()
            validation_summary_writer = tf.summary.FileWriter(checkpoint_directory_basename + '_validation')
            validation_session = tf.train.MonitoredSession()
            latest_validated_checkpoint_path = None

        print('Starting training...')
        with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_directory_basename + '_train',
                                               save_checkpoint_secs=900) as session:
            if self.settings.restore_mode == 'transfer':
                print('Restoring from {}...'.format(self.settings.restore_checkpoint_directory))
                restorer.restore(session, tf.train.latest_checkpoint(self.settings.restore_checkpoint_directory))
            self.log_source_files(checkpoint_directory_basename + '_source')
            while not session.should_stop():
                start_step_time = time.time()
                _, loss, step = session.run([training_op, loss_tensor, self.global_step])
                step_time = time.time() - start_step_time
                print('Step: {} - Loss: {:.4f} - Step Time: {:.1f}'.format(step, loss, step_time))
                # Run validation if there's a new checkpoint to validate.
                latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_directory_basename + '_train')
                if latest_checkpoint_path != latest_validated_checkpoint_path:
                    print('Running validation summaries...')
                    validation_saver.restore(validation_session, latest_checkpoint_path)
                    validation_summary_writer.add_summary(validation_session.run(validation_summaries),
                                                          global_step=step)
                    latest_validated_checkpoint_path = latest_checkpoint_path

    def train_unlabeled_gan(self):
        print('Building train graph...')
        with tf.name_scope('True'):
            self.current_data = 'True'
            true_density_error_tensor, true_count_error_tensor = self.create_network(run_type='train')
        with tf.name_scope('Generated'):
            self.current_data = 'Unlabeled'
            generated_predicted_labels_tensor, generated_predicted_count_maps_tensor = self.create_generated_network()
            generated_predicted_counts_tensor = self.example_mean_pixel_sum(generated_predicted_count_maps_tensor)
            generated_predicted_density_tensor = self.example_mean_pixel_sum(generated_predicted_labels_tensor)
            generated_predicted_absolute_tensor = self.example_mean_pixel_sum(tf.abs(generated_predicted_labels_tensor))
            generator_counts_to_negative_average = tf.abs(
                tf.add(generated_predicted_counts_tensor, self.average_train_count))
            generator_labels_to_negative_average = tf.abs(
                tf.add(generated_predicted_density_tensor, self.average_train_count))
        with tf.name_scope('Unlabeled'):
            self.current_data = 'Generated'
            unlabeled_predicted_labels_tensor, unlabeled_predicted_count_maps_tensor = self.create_unlabeled_inference_network()
            unlabeled_counts = tf.reduce_sum(unlabeled_predicted_count_maps_tensor, axis=[1, 2, 3])
            unlabeled_density_sums = tf.reduce_sum(unlabeled_predicted_labels_tensor, axis=[1, 2, 3])
            unlabeled_predicted_counts_tensor = self.example_mean_pixel_sum(unlabeled_predicted_count_maps_tensor)
            unlabeled_predicted_density_tensor = self.example_mean_pixel_sum(unlabeled_predicted_labels_tensor)

            count_below_loss = tf.maximum(tf.subtract(tf.divide(self.lookup_dictionary['guesses'], 2), unlabeled_counts), 0)
            count_above_loss = tf.maximum(tf.subtract(unlabeled_counts, tf.multiply(self.lookup_dictionary['guesses'], 2)), 0)
            label_below_loss = tf.maximum(tf.subtract(tf.divide(self.lookup_dictionary['guesses'], 2), unlabeled_density_sums), 0)
            label_above_loss = tf.maximum(tf.subtract(unlabeled_density_sums, tf.multiply(self.lookup_dictionary['guesses'], 2)), 0)
            unlabeled_count_loss = tf.reduce_sum(tf.add(count_above_loss, count_below_loss))
            unlabeled_label_loss = tf.reduce_sum(tf.add(label_above_loss, label_below_loss))

            unlabeled_counts_to_guesses = tf.abs(tf.subtract(unlabeled_counts, self.lookup_dictionary['guesses']))
            unlabeled_counts_to_guess_mean = tf.reduce_mean(unlabeled_counts_to_guesses)
            unlabeled_labels_to_guesses = tf.abs(tf.subtract(unlabeled_density_sums, self.lookup_dictionary['guesses']))
            unlabeled_labels_to_guess_mean = tf.reduce_mean(unlabeled_labels_to_guesses)

        true_loss_tensor = tf.add(tf.multiply(tf.constant(self.density_to_count_loss_ratio), true_density_error_tensor),
                                  true_count_error_tensor)
        predictor_loss_tensor = tf.add(tf.multiply(tf.constant(self.density_to_count_loss_ratio),
                                                   self.lookup_dictionary['predictor_density_error_tensor']),
                                       self.lookup_dictionary['predictor_count_error_tensor'])
        discriminator_unlabeled_loss_tensor = tf.add(tf.multiply(tf.constant(self.density_to_count_loss_ratio),
                                                                 unlabeled_label_loss),
                                                     unlabeled_count_loss)
        discriminator_generated_loss_tensor = tf.add(tf.abs(tf.multiply(tf.constant(self.density_to_count_loss_ratio),
                                                                        generated_predicted_absolute_tensor)),
                                                     tf.abs(generated_predicted_counts_tensor))
        feature_matching_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(self.middle_layer_outputs['Generated'], axis=0) -
                                                      tf.reduce_mean(self.middle_layer_outputs['Unlabeled'], axis=0)))

        epsilon1 = 0.0000000001
        upl = tf.maximum(self.lookup_dictionary['unlabeled_predicted_labels'], 0.0)
        weighted_unlabeled_labels_features = tf.multiply(self.middle_layer_outputs['Unlabeled'], upl)
        weighted_unlabeled_labels_features = tf.divide(weighted_unlabeled_labels_features, tf.reduce_sum(upl, axis=[1, 2, 3], keep_dims=True) + epsilon1)
        tpl = tf.maximum(self.lookup_dictionary['true_predicted_labels'], 0.0)
        weighted_true_labels_features = tf.multiply(self.middle_layer_outputs['True'], tpl)
        weighted_true_labels_features = tf.divide(weighted_true_labels_features, tf.reduce_sum(tpl, axis=[1, 2, 3], keep_dims=True) + epsilon1)
        upcm = tf.maximum(self.lookup_dictionary['unlabeled_predicted_count_maps'], 0.0)
        weighted_unlabeled_counts_features = tf.multiply(self.middle_layer_outputs['Unlabeled'], upcm)
        weighted_unlabeled_counts_features = tf.divide(weighted_unlabeled_counts_features, tf.reduce_sum(upcm, axis=[1, 2, 3], keep_dims=True) + epsilon1)
        tpcm = tf.maximum(self.lookup_dictionary['true_predicted_count_maps'], 0.0)
        weighted_true_counts_features = tf.multiply(self.middle_layer_outputs['True'], tpcm)
        weighted_true_counts_features = tf.divide(weighted_true_counts_features, tf.reduce_sum(tpcm, axis=[1, 2, 3], keep_dims=True) + epsilon1)
        random_ratio = tf.random_uniform([])
        labels_weighted_features = tf.add(tf.multiply(random_ratio, weighted_unlabeled_labels_features),
                                   tf.multiply(tf.subtract(1.0, random_ratio), weighted_true_labels_features))
        counts_weighted_features = tf.add(tf.multiply(random_ratio, weighted_unlabeled_counts_features),
                                          tf.multiply(tf.subtract(1.0, random_ratio), weighted_true_counts_features))
        labels_feature_matching_loss = tf.reduce_mean(tf.square(
            tf.reduce_mean(self.middle_layer_outputs['Generated'], axis=[0, 1, 2]) - tf.reduce_mean(
                labels_weighted_features, axis=[0, 1, 2])))
        counts_feature_matching_loss = tf.reduce_mean(tf.square(
            tf.reduce_mean(self.middle_layer_outputs['Generated'], axis=[0, 1, 2]) - tf.reduce_mean(
                counts_weighted_features, axis=[0, 1, 2])))
        feature_matching_loss = tf.add(tf.multiply(tf.constant(self.density_to_count_loss_ratio),
                                                               labels_feature_matching_loss),
                                                   counts_feature_matching_loss)
        scaled_feature_matching_loss = tf.multiply(
            tf.constant(self.feature_matching_parameter * self.settings.image_height * self.settings.image_width, dtype=tf.float32),
            feature_matching_loss)
        generator_loss_tensor = scaled_feature_matching_loss

        with tf.name_scope('Loss'):
            tf.summary.scalar('Labels Multiplier', self.lookup_dictionary['labels_multiplier'])
            tf.summary.scalar('Counts Multiplier', self.lookup_dictionary['counts_multiplier'])
            tf.summary.scalar('Feature Matching Loss Ratio', scaled_feature_matching_loss / generator_loss_tensor)
            tf.summary.scalar('True Discriminator Loss', true_loss_tensor)
            tf.summary.scalar('Average Train Count', self.average_train_count)
            tf.summary.scalar('Mean predicted density', tf.reduce_mean(self.lookup_dictionary['true_predicted_labels']))
            tf.summary.scalar('Generated Discriminator Loss', discriminator_generated_loss_tensor)
            tf.summary.scalar('Generated Predicted Count', generated_predicted_counts_tensor)
            tf.summary.scalar('Generated Predicted Density', generated_predicted_density_tensor)
            tf.summary.scalar('Unlabeled Discriminator Loss', discriminator_unlabeled_loss_tensor)
            tf.summary.scalar('Unlabeled Predicted Count', unlabeled_predicted_counts_tensor)
            tf.summary.scalar('Unlabeled Predicted Density', unlabeled_predicted_density_tensor)
            tf.summary.scalar('Generator Loss', generator_loss_tensor)
            tf.summary.scalar('Percentage Discriminator Loss From Generated',
                              discriminator_generated_loss_tensor / (discriminator_generated_loss_tensor +
                                                                     discriminator_unlabeled_loss_tensor +
                                                                     true_loss_tensor))
            tf.summary.scalar('Percentage Discriminator Loss From Unlabeled',
                              discriminator_unlabeled_loss_tensor / (discriminator_generated_loss_tensor +
                                                                     discriminator_unlabeled_loss_tensor +
                                                                     true_loss_tensor))
        input_penalty_epsilons = tf.random_uniform([3])
        input_penalty_epsilons = tf.divide(input_penalty_epsilons, tf.reduce_sum(input_penalty_epsilons))
        penalty_examples = tf.add_n([tf.multiply(self.lookup_dictionary['labeled_images_tensor'],
                                                 input_penalty_epsilons[0]),
                                     tf.multiply(self.lookup_dictionary['unlabeled_images_tensor'],
                                                 input_penalty_epsilons[1]),
                                     tf.multiply(self.lookup_dictionary['generated_images_tensor'],
                                                 input_penalty_epsilons[2])])
        dropout_arg_scope = tf.contrib.framework.arg_scope([tf.contrib.layers.dropout],
                                                           keep_prob=0.5)
        with tf.variable_scope('inference', reuse=True), dropout_arg_scope:
            penalty_tensor_list = self.create_experimental_inference_op(penalty_examples)
        penalty_gradients = tf.gradients([*penalty_tensor_list], penalty_examples)
        gradient_penalty = 10.0 * tf.square(tf.norm(penalty_gradients, ord=2) - 1.0)
        optimizer = tf.train.AdamOptimizer(self.learning_rate_tensor)
        discriminator_compute_op = optimizer.compute_gradients(
            tf.add_n([true_loss_tensor, discriminator_generated_loss_tensor, discriminator_unlabeled_loss_tensor,
                      gradient_penalty]),
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='inference')
        )
        discriminator_gradients = [tf.reshape(pair[0], [-1]) for pair in discriminator_compute_op
                                   if 'weights:' in pair[1].name]
        tf.summary.scalar('Discriminator mean gradient', tf.reduce_mean(tf.abs(tf.concat(discriminator_gradients,
                                                                                         axis=0))))
        predictor_compute_op = optimizer.compute_gradients(
            predictor_loss_tensor,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='True/predictor')
        )
        generator_compute_op = optimizer.compute_gradients(
            generator_loss_tensor,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        )
        generator_gradients = [tf.reshape(pair[0], [-1]) for pair in generator_compute_op
                               if 'weights:' in pair[1].name]
        tf.summary.scalar('Generator mean gradient', tf.reduce_mean(tf.abs(tf.concat(generator_gradients, axis=0))))
        both_training_op = optimizer.apply_gradients(
            generator_compute_op + discriminator_compute_op + predictor_compute_op, global_step=self.global_step)
        discriminator_training_op = optimizer.apply_gradients(discriminator_compute_op + predictor_compute_op,
                                                              global_step=self.global_step)
        #discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='inference')
        #clip_ops = []
        #for variable in discriminator_variables:
        #    if 'weights:' in variable.name:
        #        clip_ops.append(tf.clip_by_value(variable, -self.clip_value, self.clip_value))
        #clip_weights_op = tf.group(*clip_ops)
        checkpoint_directory_basename = self.get_checkpoint_directory_basename()
        if self.settings.restore_mode == 'transfer':
            restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        else:
            restorer = None

        print('Building validation graph...')
        validation_graph = tf.Graph()
        with validation_graph.as_default(), tf.device('/cpu:0'):
            with tf.name_scope('True'):
                self.create_network(run_type='validation')
                validation_summaries = tf.summary.merge_all()
                validation_saver = tf.train.Saver()
                validation_summary_writer = tf.summary.FileWriter(checkpoint_directory_basename + '_validation')
                validation_session = tf.train.MonitoredSession()
                latest_validated_checkpoint_path = None

        print('Starting training...')
        with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_directory_basename + '_train',
                                               save_checkpoint_secs=900) as session:
            if self.settings.restore_mode == 'transfer':
                print('Restoring from {}...'.format(self.settings.restore_checkpoint_directory))
                restorer.restore(session, tf.train.latest_checkpoint(self.settings.restore_checkpoint_directory))
            self.log_source_files(checkpoint_directory_basename + '_source')
            step = 0
            while not session.should_stop():
                if step % self.generator_train_step_period == 0:
                    training_op = both_training_op
                else:
                    training_op = discriminator_training_op
                start_step_time = time.time()
                _, loss, step = session.run([training_op, true_loss_tensor, self.global_step])
                #session.run([clip_weights_op])
                step_time = time.time() - start_step_time
                print('Step: {} - Loss: {:.4f} - Step Time: {:.1f}'.format(step, loss, step_time))
                # Run validation if there's a new checkpoint to validate.
                latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_directory_basename + '_train')
                if latest_checkpoint_path != latest_validated_checkpoint_path:
                    print('Running validation summaries...')
                    validation_saver.restore(validation_session, latest_checkpoint_path)
                    validation_summary_writer.add_summary(validation_session.run(validation_summaries),
                                                          global_step=step)
                    latest_validated_checkpoint_path = latest_checkpoint_path

    def test(self):
        """
        Runs the testing of the network.
        """
        # print('Building testing graph...')
        self.settings.batch_size = 1
        self.create_network(run_type='test')
        labels_tensor = self.lookup_dictionary['labels_tensor']
        predicted_labels_tensor = self.lookup_dictionary['predicted_labels_tensor']
        predicted_counts_tensor = self.lookup_dictionary['predicted_counts_tensor']
        saver = tf.train.Saver()
        total_count = 0
        total_predicted_count = 0
        total_count_error = 0
        number_of_examples = 0
        total_density_count_error = 0
        # print('Running test...')
        with tf.train.MonitoredSession() as session:
            latest_checkpoint_path = tf.train.latest_checkpoint(self.get_checkpoint_directory_basename())
            saver.restore(session, latest_checkpoint_path)
            while not session.should_stop():
                label, predicted_count, predicted_labels = session.run(
                    [labels_tensor, predicted_counts_tensor, predicted_labels_tensor])
                count = np.sum(label)
                predicted_density_count = np.sum(predicted_labels)
                total_count += count
                total_predicted_count += predicted_count
                total_count_error += np.abs(count - predicted_count)
                total_density_count_error += np.abs(count - predicted_density_count)
                number_of_examples += 1
                print('{} examples processed'.format(number_of_examples), end='\r')
        # print('Total count: {}'.format(total_count))
        # print('Total predicted count: {}'.format(total_predicted_count))
        # print('Total count error: {}'.format(total_count_error))
        # print('Total density count error: {}'.format(total_density_count_error))
        print('')
        if 'single_camera' in self.settings.datasets_json:
            print('Scene {}'.format(self.settings.datasets_json.replace('single_camera_', '')))
        else:
            print('Validation' if self.settings.test_validation_swap else 'Test')
        print('Average count error: {}'.format(total_count_error / number_of_examples))
        print('Average density count error: {}'.format(total_density_count_error / number_of_examples))
        print('')


if __name__ == '__main__':
    test_settings = Settings()
    if test_settings.run_mode == 'test':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        settings_list = []
        validation_settings = Settings()
        validation_settings.test_validation_swap = True
        settings_list.append(validation_settings)
        settings_list.append(test_settings)
        for index in range(1, 6):
            single_camera_settings = Settings()
            single_camera_settings.datasets_json = 'single_camera_{}.json'.format(index)
            settings_list.append(single_camera_settings)
        for settings in settings_list:
            crowd_net = CrowdNet(settings)
            crowd_net.test()
            tf.reset_default_graph()
    else:
        crowd_net = CrowdNet()
        crowd_net.train_unlabeled_gan()
