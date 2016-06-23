"""
Code related to the CrowdNet.
"""
import tensorflow as tf

from crowd_data import CrowdData
from go_net import GoNet
from interface import Interface
from convenience import weight_variable, bias_variable, leaky_relu, conv2d, size_from_stride_two


class CrowdNet(GoNet):
    """
    A neural network class to estimate crowd density from single 2D images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = CrowdData()

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
        self.create_person_count_summaries(labels, predicted_labels)
        return absolute_differences_tensor

    def create_person_count_summaries(self, labels, predicted_labels):
        """
        Creates the summaries for the counts of people.

        :param labels: The true person density labels.
        :type labels: tf.Tensor
        :param predicted_labels: The predicted person density labels.
        :type predicted_labels: tf.Tensor
        """
        true_person_count_tensor = self.mean_person_count_for_labels(labels)
        predicted_person_count_tensor = self.mean_person_count_for_labels(predicted_labels)
        person_miscount_tensor = tf.abs(true_person_count_tensor - predicted_person_count_tensor)
        relative_person_miscount_tensor = person_miscount_tensor / true_person_count_tensor
        tf.scalar_summary('True person count', true_person_count_tensor)
        tf.scalar_summary('Predicted person count', predicted_person_count_tensor)
        tf.scalar_summary('Person miscount', person_miscount_tensor)
        tf.scalar_summary('Relative person miscount', relative_person_miscount_tensor)

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

    def create_inference_op(self, images):
        """
        Creates and adds graph components to perform a forward pass estimating label maps from RGB images.
        Overrides the GoNet method of the same name.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        return self.create_patchwise_inference_op(images)

    def create_standard_net_inference_op(self, images):
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
            fc0_size = size_from_stride_two(self.data.image_height, iterations=2) * size_from_stride_two(
                self.data.image_width, iterations=2) * 32
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
            fc3_size = self.data.image_height * self.data.image_width
            w_fc = weight_variable([fc2_size, fc3_size])
            b_fc = bias_variable([fc3_size])

            h_fc = leaky_relu(tf.matmul(h_fc, w_fc) + b_fc)
            predicted_labels = tf.reshape(h_fc, [-1, self.data.image_height, self.data.image_width, 1])

        return predicted_labels

    def create_patchwise_inference_op(self, images):
        """
        Performs a forward pass estimating label maps from RGB images using a patchwise graph setup.

        :param images: The RGB images tensor.
        :type images: tf.Tensor
        :return: The label maps tensor.
        :rtype: tf.Tensor
        """
        with tf.name_scope('conv1'):
            w_conv = weight_variable([3, 3, 3, 32])
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


if __name__ == '__main__':
    interface = Interface(network_class=CrowdNet)
    interface.train()
