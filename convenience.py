"""
Code for simplifying generically used functions.
"""
import tensorflow as tf


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


def leaky_relu(x):
    """
    A basic implementation of a leaky ReLU.

    :param x: The input of the ReLU activation.
    :type x: tf.Tensor
    :return: The tensor filtering on the leaky activation.
    :rtype: tf.Tensor
    """
    return tf.maximum(tf.mul(0.001, x), x)