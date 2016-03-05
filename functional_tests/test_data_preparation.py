"""
Tests the data preparation scripts.
"""
import os
import numpy as np

from DataPreparation import convert_mat_file_to_numpy_file


class TestDataPreparation:
    def test_can_convert_from_mat_file_to_numpy_files(self):
        # Clean up a previous creation.
        images_numpy_file_path = os.path.join('data', 'nyud_images.npy')
        try:
            os.remove(images_numpy_file_path)
        except OSError:
            pass
        depths_numpy_file_path = os.path.join('data', 'nyud_depths.npy')
        try:
            os.remove(depths_numpy_file_path)
        except OSError:
            pass

        # Run the conversion script.
        mat_file_path = os.path.join('data', 'nyu_depth_v2_labeled.mat')
        convert_mat_file_to_numpy_file(mat_file_path)

        # Check that the files are created.
        assert os.path.isfile(images_numpy_file_path)
        assert os.path.isfile(depths_numpy_file_path)

        # Check that magic values are correct when the data is reloaded from numpy files.
        images = np.load(images_numpy_file_path)
        assert images[10, 10, 1, 10] == 89
        depths = np.load(depths_numpy_file_path)
        assert depths[10, 10, 10] == 2.27751
