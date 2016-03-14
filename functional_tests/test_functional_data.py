"""
Tests the data preparation scripts.
"""
import os
import math
import numpy as np
import tensorflow as tf
from pytest import fail

from data_preparation import DataPreparation
from data import Data


class TestFunctionalData:
    def test_can_convert_from_mat_file_to_numpy_files(self):
        # Prepare paths.
        images_numpy_file_path = os.path.join('functional_tests', 'test_data', 'images_nyud_micro.npy')
        depths_numpy_file_path = os.path.join('functional_tests', 'test_data', 'depths_nyud_micro.npy')
        mat_file_path = os.path.join('functional_tests', 'test_data', 'nyud_micro.mat')

        # Run the conversion script.
        DataPreparation().convert_mat_file_to_numpy_file(mat_file_path)

        # Check that the files are created.
        assert os.path.isfile(images_numpy_file_path)
        assert os.path.isfile(depths_numpy_file_path)

        # Check that magic values are correct when the data is reloaded from numpy files.
        images = np.load(images_numpy_file_path)
        assert images[5, 10, 10, 1] == 91
        depths = np.load(depths_numpy_file_path)
        assert math.isclose(depths[5, 10, 10], 3.75686, abs_tol=0.001)

        # Clean up.
        remove_file_if_exists(images_numpy_file_path)
        remove_file_if_exists(depths_numpy_file_path)

    def test_can_convert_from_mat_to_tfrecord_and_read_tfrecord(self):
        # Prepare paths.
        data_directory = os.path.join('functional_tests', 'test_data')
        mat_file_path = os.path.join(data_directory, 'nyud_micro.mat')
        tfrecords_file_path = os.path.join(data_directory, 'nyud_micro.train.tfrecords')

        # Run the conversion script.
        DataPreparation().convert_mat_to_tfrecord(mat_file_path)

        # Check that the file is created.
        assert os.path.isfile(tfrecords_file_path)

        # Reload data.
        data = Data(data_directory=data_directory, data_name='nyud_micro')
        images, depths = data.inputs(data_type='train', batch_size=10)

        # Check that magic values are correct when the data is reloaded.
        magic_image_numbers = [-0.17450979, -0.15882352, -0.15490195, -0.15098038, -0.14705881,
                               -0.14313725, -0.11960781, -0.056862712, 0.0058823824]
        magic_depth_numbers = [1.1285654, 1.8865139, 2.104018, 2.1341071, 2.6960645,
                               3.318316, 3.4000545, 3.4783292, 3.7568643, 3.9500945]
        session = tf.Session()
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)
        try:
            with session.as_default():
                assert np.isclose(magic_image_numbers, images.eval()[5, 10, 10, 1], atol=0.00001).any()
                assert np.isclose(magic_depth_numbers, depths.eval()[5, 10, 10], atol=0.00001).any()
        except tf.errors.OutOfRangeError:
            fail('Should not hit this.')
        finally:
            coordinator.request_stop()
        coordinator.join(threads)
        session.close()

        # Clean up.
        remove_file_if_exists(tfrecords_file_path)


def remove_file_if_exists(file_path):
    try:
        os.remove(file_path)
    except OSError:
        pass
