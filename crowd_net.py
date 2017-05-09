"""
Code related to the CrowdNet.
"""
import datetime
import tensorflow as tf
import os
import numpy as np
import time

from gonet.net import Net
from gonet.interface import Interface
from gonet.convenience import leaky_relu

from crowd_data import CrowdData
from settings import Settings


class CrowdNet(Net):
    """
    A neural network class to estimate crowd density from single 2D images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(settings=Settings(), *args, **kwargs)

        self.density_to_count_loss_ratio = 20.0
        self.data = CrowdData()

        self.histograms_on = False
        self.alternate_loss_on = True
        self.edge_percentage = 0.0

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

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.learning_rate_tensor = tf.train.exponential_decay(self.settings.initial_learning_rate,
                                                               self.global_step,
                                                               self.settings.learning_rate_decay_steps,
                                                               self.settings.learning_rate_decay_rate)

    @staticmethod
    def create_experimental_inference_op(images):
        """
        Performs a forward pass estimating label maps from RGB images using a patchwise graph setup.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d],
                                            padding='SAME',
                                            normalizer_fn=tf.contrib.layers.batch_norm,
                                            activation_fn=leaky_relu,
                                            kernel_size=3):
            module1_output = tf.contrib.layers.conv2d(inputs=images, num_outputs=32)
            module2_output = tf.contrib.layers.conv2d(inputs=module1_output, num_outputs=64)
            module3_output = tf.contrib.layers.conv2d(inputs=module2_output, num_outputs=64)
            module4_output = tf.contrib.layers.conv2d(inputs=module3_output, num_outputs=128)
            module5_output = tf.contrib.layers.conv2d(inputs=module4_output, num_outputs=128)
            module6_output = tf.contrib.layers.conv2d(inputs=module5_output, num_outputs=256)
            module7_output = tf.contrib.layers.conv2d(inputs=module6_output, num_outputs=256)
            module8_output = tf.contrib.layers.conv2d(inputs=module7_output, num_outputs=10, kernel_size=1)
            module9_output = tf.contrib.layers.conv2d(inputs=module8_output, num_outputs=10, kernel_size=1,
                                                      normalizer_fn=None)
            person_density_output = tf.contrib.layers.conv2d(inputs=module9_output, num_outputs=1, kernel_size=1,
                                                             activation_fn=None, normalizer_fn=None)
            person_count_map_output = tf.contrib.layers.conv2d(inputs=module9_output, num_outputs=1, kernel_size=1,
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

        true_person_count_tensor = self.example_mean_pixel_sum(labels_tensor)
        count_error_tensor = tf.abs(tf.subtract(true_person_count_tensor, predicted_counts_tensor))

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
        training_op = tf.train.AdamOptimizer(self.learning_rate_tensor).minimize(value_to_minimize,
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
            images_tensor, labels_tensor = self.data.create_input_tensors_for_dataset(
                data_type=run_type,
                batch_size=self.settings.batch_size
            )

        with tf.variable_scope('inference'):
            predicted_labels_tensor, predicted_count_maps_tensor = self.create_experimental_inference_op(images_tensor)

            masked_tensors = self.apply_roi_mask(labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor)
            labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor = masked_tensors

            predicted_counts_tensor = self.example_mean_pixel_sum(predicted_count_maps_tensor)

        density_error_tensor, count_error_tensor = self.create_error_tensors(labels_tensor,
                                                                             predicted_labels_tensor,
                                                                             predicted_counts_tensor)

        self.density_comparison_summary(images_tensor, labels_tensor, predicted_labels_tensor)

        self.lookup_dictionary['labels_tensor'] = labels_tensor
        self.lookup_dictionary['predicted_labels_tensor'] = predicted_labels_tensor
        self.lookup_dictionary['predicted_counts_tensor'] = predicted_counts_tensor

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
        if self.settings.restore_checkpoint_directory and self.settings.restore_mode == 'continue':
            return os.path.join(self.settings.logs_directory, self.settings.restore_checkpoint_directory)
        elif self.settings.run_mode == 'test':
            return os.path.join(self.settings.logs_directory, self.settings.restore_checkpoint_directory + '_train')
        else:
            return os.path.join(self.settings.logs_directory, self.settings.network_name + ' ' +
                                datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))

    def unlabeled_generator(self):
        assert self.settings.image_height is 144 and self.settings.image_width is 180
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.conv2d_transpose],
                                            padding='SAME',
                                            normalizer_fn=tf.contrib.layers.batch_norm,
                                            activation_fn=leaky_relu,
                                            kernel_size=5):
            noise = tf.random_uniform([self.settings.batch_size, 1, 1, 50])
            net = tf.contrib.layers.conv2d_transpose(noise, 1024, kernel_size=[4, 5], stride=1, padding='VALID')
            net = tf.contrib.layers.conv2d_transpose(net, 512, stride=3)
            net = tf.contrib.layers.conv2d_transpose(net, 256, stride=3)
            net = tf.contrib.layers.conv2d_transpose(net, 128, stride=2)
            net = tf.contrib.layers.conv2d_transpose(net, 3, stride=2, activation_fn=tf.tanh, normalizer_fn=None)
            mean, variance = tf.nn.moments(net, axes=[1, 2, 3], keep_dims=True)
            images_tensor = (net - mean) / tf.sqrt(variance)
        return images_tensor

    def create_generated_network(self):
        with tf.variable_scope('generator'):
            images_tensor = self.unlabeled_generator()
            tf.summary.image('Generated Images', images_tensor)

        with tf.variable_scope('inference', reuse=True):
            predicted_labels_tensor, predicted_count_maps_tensor = self.create_experimental_inference_op(images_tensor)
            predicted_counts_tensor = self.example_mean_pixel_sum(predicted_count_maps_tensor)
            predicted_total_density_tensor = self.example_mean_pixel_sum(predicted_count_maps_tensor)

        return predicted_total_density_tensor, predicted_counts_tensor

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
                                               save_checkpoint_secs=180) as session:
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
            true_density_error_tensor, true_count_error_tensor = self.create_network(run_type='train')
        with tf.name_scope('Generated'):
            generated_predicted_density_tensor, generated_predicted_counts_tensor = self.create_generated_network()
        true_loss_tensor = tf.add(tf.multiply(tf.constant(self.density_to_count_loss_ratio), true_density_error_tensor),
                                  true_count_error_tensor)
        discriminator_generated_loss_tensor = tf.add(tf.abs(tf.multiply(tf.constant(self.density_to_count_loss_ratio),
                                                                        generated_predicted_density_tensor)),
                                                     tf.abs(generated_predicted_counts_tensor))
        generator_loss_tensor = tf.negative(tf.add(tf.multiply(tf.constant(self.density_to_count_loss_ratio),
                                                               generated_predicted_density_tensor),
                                                   generated_predicted_counts_tensor))
        with tf.name_scope('Loss'):
            tf.summary.scalar('True Discriminator Loss', true_loss_tensor)
            tf.summary.scalar('Generated Discriminator Loss', discriminator_generated_loss_tensor)
            tf.summary.scalar('Generator Loss', generator_loss_tensor)
        optimizer = tf.train.AdamOptimizer(self.learning_rate_tensor)
        discriminator_compute_op = optimizer.compute_gradients(
            tf.add(true_loss_tensor, discriminator_generated_loss_tensor),
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='inference')
        )
        generator_compute_op = optimizer.compute_gradients(
            generator_loss_tensor,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        )
        training_op = optimizer.apply_gradients(discriminator_compute_op + generator_compute_op,
                                                global_step=self.global_step)
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
            while not session.should_stop():
                start_step_time = time.time()
                _, loss, step = session.run([training_op, true_loss_tensor, self.global_step])
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
        print('Building testing graph...')
        self.settings.batch_size = 1
        self.create_network(run_type='test')
        labels_tensor = self.lookup_dictionary['labels_tensor']
        predicted_labels_tensor = self.lookup_dictionary['predicted_labels_tensor']
        predicted_counts_tensor = self.lookup_dictionary['predicted_counts_tensor']
        total_count = 0
        total_predicted_count = 0
        total_count_error = 0
        number_of_examples = 0
        total_density_count_error = 0
        print('Running test...')
        with tf.train.MonitoredTrainingSession(checkpoint_dir=self.get_checkpoint_directory_basename(),
                                               save_checkpoint_secs=None, save_summaries_steps=None) as session:
            while not session.should_stop():
                label, predicted_count, predicted_labels = session.run([labels_tensor, predicted_counts_tensor, predicted_labels_tensor])
                count = np.sum(label)
                predicted_density_count = np.sum(predicted_labels)
                total_count += count
                total_predicted_count += predicted_count
                total_count_error += np.abs(count - predicted_count)
                total_density_count_error += np.abs(count - predicted_density_count)
                number_of_examples += 1
                print('{} examples processed'.format(number_of_examples), end='\r')
        print('Total count: {}'.format(total_count))
        print('Total predicted count: {}'.format(total_predicted_count))
        print('Total count error: {}'.format(total_count_error))
        print('Total density count error: {}'.format(total_density_count_error))
        print('Average count error: {}'.format(total_count_error / number_of_examples))
        print('Average density count error: {}'.format(total_density_count_error / number_of_examples))


if __name__ == '__main__':
    crowd_net = CrowdNet()
    crowd_net.train_unlabeled_gan()
