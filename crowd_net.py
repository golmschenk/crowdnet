"""
Code related to the CrowdNet.
"""
import tensorflow as tf

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
        absolute_differences_tensor = self.create_absolute_differences_tensor(predicted_labels, labels)
        if self.edge_percentage > 0.001:
            edge_width = int(self.settings.image_width * self.edge_percentage)
            edge_height = int(self.settings.image_height * self.edge_percentage)
            absolute_differences_tensor = absolute_differences_tensor[:, edge_height:-edge_height,
                                          edge_width:-edge_width]
            padding = [[0, 0], [edge_height, edge_height], [edge_width, edge_width], [0, 0]]
            absolute_differences_tensor = tf.pad(absolute_differences_tensor, padding)

        relative_person_miscount_tensor = self.create_person_count_summaries(labels, predicted_labels)
        if self.alternate_loss_on:
            self.alternate_loss = relative_person_miscount_tensor
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
        true_person_count_tensor = self.mean_person_count_for_labels(labels)
        predicted_person_count_tensor = self.mean_person_count_for_labels(predicted_labels)
        person_miscount_tensor = tf.abs(true_person_count_tensor - predicted_person_count_tensor)
        relative_person_miscount_tensor = person_miscount_tensor / true_person_count_tensor
        tf.summary.scalar('True person count', true_person_count_tensor)
        tf.summary.scalar('Predicted person count', predicted_person_count_tensor)
        tf.summary.scalar('Person miscount', person_miscount_tensor)
        tf.summary.scalar('Relative person miscount', relative_person_miscount_tensor)
        return relative_person_miscount_tensor

    @staticmethod
    def mean_person_count_for_labels(labels_tensor):
        """
        Sums the labels per image and takes the mean over the images.

        :param labels_tensor: The person density labels tensor to process.
        :type labels_tensor: tf.Tensor
        :return: The mean count tensor.
        :rtype: tf.Tensor
        """
        mean_person_count_tensor = tf.reduce_mean(tf.reduce_sum(labels_tensor, [1, 2]))
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

    def create_training_op(self, value_to_minimize):
        """
        Create and add the training op to the graph.

        :param value_to_minimize: The value to train on.
        :type value_to_minimize: tf.Tensor or list[tf.Tensor]
        :return: The training op.
        :rtype: tf.Operation
        """
        if self.alternate_loss_on:
            value_to_minimize = tf.cond(tf.equal(tf.mod(self.global_step, 2), 0),
                                        lambda: value_to_minimize,
                                        lambda: self.alternate_loss)
        return super().create_training_op(value_to_minimize)

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


if __name__ == '__main__':
    interface = Interface(network_class=CrowdNet)
    interface.run()
