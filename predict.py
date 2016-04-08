"""
Code for predicting for a single example.
"""
import numpy as np
import tensorflow as tf
import os


class Predictor:
    """
    A class for predicting a single example based on an existing model.
    """
    def __init__(self, graph_def_file_path, variables_file_path):
        self.graph_def_file_path = graph_def_file_path
        self.variables_file_path = variables_file_path

    def predict(self, example):
        """
        Predicts the label for a given example.

        :param example: The example to do the prediction on.
        :type example: np.ndarray
        :return: The prediction.
        :rtype: np.ndarray
        """
        with tf.gfile.FastGFile(self.graph_def_file_path, 'rb') as graph_def_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_def_file.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, self.variables_file_path)

            example_tensor = tf.expand_dims(example, 0)
            predicted_depths = session.graph.get_tensor_by_name('predicted_depths')
            images = session.graph.get_tensor_by_name('images')

            predicted_depths_output = session.run(predicted_depths, feed_dict={images: example_tensor})
            predicted_depth = tf.squeeze(predicted_depths_output)
            return predicted_depth

    def save_predict_for_numpy_file(self, numpy_file_path):
        """
        Creates a prediction numpy for the example in the given numpy file. The prediction is saved next to the
        original numpy file.

        :param numpy_file_path: The path to the example numpy file.
        :type numpy_file_path: str
        """
        sample = np.load(numpy_file_path)
        prediction = self.predict(sample)
        name, ext = os.path.splitext(numpy_file_path)
        np.save(name + _prediction + ext, prediction)
