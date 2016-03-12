"""
Tests the data preparation scripts.
"""
import os

import math
import numpy as np

from data_preparation import DataPreparation


class TestDataPreparation:
    def test_can_convert_from_mat_file_to_numpy_files(self):
        # Clean up a previous creation.
        images_numpy_file_path = os.path.join('functional_tests', 'test_data', 'images_nyud_micro.npy')
        depths_numpy_file_path = os.path.join('functional_tests', 'test_data', 'depths_nyud_micro.npy')

        def remove_file_if_exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

        remove_file_if_exists(images_numpy_file_path)
        remove_file_if_exists(depths_numpy_file_path)

        # Run the conversion script.
        mat_file_path = os.path.join('functional_tests', 'test_data', 'nyud_micro.mat')
        print(os.path.abspath(mat_file_path))
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
