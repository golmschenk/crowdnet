"""
Unit tests for the data_preparation module.
"""
from unittest.mock import patch, Mock
import os
import numpy as np

from data_preparation import DataPreparation


class TestDataPreparation:
    @patch('numpy.save')
    @patch('h5py.File')
    def test_convert_mat_file_to_numpy_file_reads_the_mat_file(self, h5py_file_mock, mock_numpy_save):
        mat_file_name = 'fake name'
        data_preparation = DataPreparation()

        data_preparation.convert_mat_file_to_numpy_file(mat_file_name)

        assert h5py_file_mock.call_args == ((mat_file_name, 'r'),)

    @patch('numpy.save')
    @patch('h5py.File')
    def test_convert_mat_file_to_numpy_file_calls_extract_mat_data_to_numpy_array(self, h5py_file_mock,
                                                                                  mock_numpy_save):
        h5py_file_mock.return_value = 'fake mat data'
        data_preparation = DataPreparation()
        data_preparation.convert_mat_data_to_numpy_array = Mock()

        data_preparation.convert_mat_file_to_numpy_file('')

        assert data_preparation.convert_mat_data_to_numpy_array.call_args_list[1] == (('fake mat data', 'depths'),)
        assert data_preparation.convert_mat_data_to_numpy_array.call_args_list[0] == (('fake mat data', 'images'),)

    def test_convert_mat_data_to_numpy_array_extracts_and_tranposes_the_data(self):
        data_preparation = DataPreparation()
        mock_mat_data = Mock()
        mock_mat_data.get.return_value = np.array([[1, 2, 3]])

        transposed_array = data_preparation.convert_mat_data_to_numpy_array(mock_mat_data, 'fake variable')

        assert mock_mat_data.get.call_args == (('fake variable',),)
        assert np.array_equal(transposed_array, np.array([[1], [2], [3]]))

    @patch('h5py.File')
    @patch('numpy.save')
    def test_convert_mat_file_to_numpy_file_writes_extracted_numpys_to_files(self, mock_numpy_save, h5py_file_mock):
        data_preparation = DataPreparation()
        data_preparation.convert_mat_data_to_numpy_array = Mock(side_effect=[1, 2])

        data_preparation.convert_mat_file_to_numpy_file('')

        assert mock_numpy_save.call_args_list[0] == ((os.path.join('data', 'images_') + '.npy', 1),)
        assert mock_numpy_save.call_args_list[1] == ((os.path.join('data', 'depths_') + '.npy', 2),)

    def test_convert_mat_data_to_numpy_array_can_specify_the_number_of_images_to_extract(self):
        data_preparation = DataPreparation()
        mock_mat_data = Mock()
        mock_mat_data.get.return_value = np.array([[1], [2], [3]])

        transposed_array = data_preparation.convert_mat_data_to_numpy_array(mock_mat_data, 'fake variable',
                                                                            number_of_samples=2)

        assert np.array_equal(transposed_array, np.array([[1, 2]]))
