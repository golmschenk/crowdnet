"""
Code related to the CrowdNet.
"""
import datetime
import tensorflow as tf
import os
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
        with tf.name_scope('inputs'):
            images_tensor, labels_tensor = self.data.create_input_tensors_for_dataset(
                data_type=run_type,
                batch_size=self.settings.batch_size
            )

        with tf.variable_scope('inference'), tf.name_scope(run_type):
            predicted_labels_tensor, predicted_count_maps_tensor = self.create_experimental_inference_op(images_tensor)

            masked_tensors = self.apply_roi_mask(labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor)
            labels_tensor, predicted_labels_tensor, predicted_count_maps_tensor = masked_tensors

            predicted_counts_tensor = self.example_mean_pixel_sum(predicted_count_maps_tensor)

        density_error_tensor, count_error_tensor = self.create_loss_tensors(labels_tensor,
                                                                            predicted_labels_tensor,
                                                                            predicted_counts_tensor)

        if self.image_summary_on:
            self.density_comparison_summary(images_tensor, labels_tensor, predicted_labels_tensor)

        return density_error_tensor, count_error_tensor

    def train(self):
        print('Building train graph...')
        train_density_error_tensor, train_count_error_tensor = self.create_network(run_type='train')
        loss_tensor = tf.add(train_density_error_tensor, tf.multiply(tf.constant(2.0), train_count_error_tensor))
        training_op = self.create_training_op(loss_tensor)
        checkpoint_directory = os.path.join(self.settings.logs_directory, self.settings.network_name + ' ' +
                                            datetime.datetime.now().strftime("y%Y_m%m_d%d_h%H_m%M_s%S"))

        print('Building validation graph...')
        validation_graph = tf.Graph()
        with validation_graph.as_default(), tf.device('/cpu:0'):
            self.create_network(run_type='validation')
            validation_summaries = tf.summary.merge_all()
            validation_saver = tf.train.Saver()
            validation_summary_writer = tf.summary.FileWriter(checkpoint_directory + '_validation')
            validation_session = tf.train.MonitoredSession()
            latest_validated_checkpoint_path = None

        print('Starting training...')
        with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_directory + '_train',
                                               save_checkpoint_secs=300) as session:
            while True:
                start_step_time = time.time()
                _, loss, step = session.run([training_op, loss_tensor, self.global_step])
                step_time = time.time() - start_step_time()
                print('Step: {} - Loss: {:.4f} - Step Time: {:.3f}'.format(step, loss, step_time))
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
