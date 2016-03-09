"""
Code for preparing data.
"""
import os
import h5py
import numpy as np


class DataPreparation:
    """
    A class for data preparation.
    """
    def convert_mat_file_to_numpy_file(self, mat_file_path):
        """
        Generate image and depth numpy files from the passed mat file path.

        :param mat_file_path: The path to the mat file.
        :type mat_file_path: str
        """
        mat_data = h5py.File(mat_file_path, 'r')
        images = self.convert_mat_data_to_numpy_array(mat_data, 'images')
        depths = self.convert_mat_data_to_numpy_array(mat_data, 'depths')
        basename = os.path.basename(os.path.splitext(mat_file_path)[0])
        np.save(os.path.join('data', 'images_' + basename) + '.npy', images)
        np.save(os.path.join('data', 'depths_' + basename) + '.npy', depths)

    @staticmethod
    def convert_mat_data_to_numpy_array(mat_data, variable_name_in_mat_data):
        """
        Converts a mat data variable to a numpy array.

        :param mat_data: The mat data containing the variable to be converted.
        :type mat_data: h5py.File
        :param variable_name_in_mat_data: The name of the variable to extract.
        :type variable_name_in_mat_data: str
        :return: The numpy array.
        :rtype: np.ndarray
        """
        mat_variable = mat_data.get(variable_name_in_mat_data)
        return np.array(mat_variable).transpose()
