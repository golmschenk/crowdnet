"""
Code for preparing data.
"""
import h5py


class DataPreparation:
    """
    A class for data preparation.
    """
    @staticmethod
    def convert_mat_file_to_numpy_file(mat_file_path):
        """
        Generate image and depth numpy files from the passed mat file path.

        :param mat_file_path: The path to the mat file.
        :type mat_file_path: str
        """
        h5py.File(mat_file_path, 'r')
