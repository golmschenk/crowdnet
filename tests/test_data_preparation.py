"""
Unit tests for the data_preparation module.
"""
from unittest.mock import patch

from data_preparation import DataPreparation


class TestDataPreparation:
    @patch('h5py.File')
    def test_convert_mat_file_to_numpy_file_reads_the_mat_file(self, h5py_file_mock):
        mat_file_name = 'fake name'
        DataPreparation.convert_mat_file_to_numpy_file(mat_file_name)

        assert h5py_file_mock.call_args == ((mat_file_name, 'r'), )
