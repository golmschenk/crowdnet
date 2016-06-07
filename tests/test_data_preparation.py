"""
Unit tests for the go_data module.
"""
from unittest.mock import patch, Mock
import os
import numpy as np

from go_data import GoData


class TestData:
    @patch('numpy.save')
    @patch('h5py.File')
    def test_convert_mat_file_to_numpy_file_reads_the_mat_file(self, h5py_file_mock, mock_numpy_save):
        mat_file_name = 'fake name'
        go_data = GoData()
        go_data.crop_data = Mock()
        go_data.convert_mat_data_to_numpy_array = Mock()

        go_data.convert_mat_file_to_numpy_file(mat_file_name)

        assert h5py_file_mock.call_args == ((mat_file_name, 'r'),)

    @patch('numpy.save')
    @patch('h5py.File')
    def test_convert_mat_file_to_numpy_file_calls_extract_mat_data_to_numpy_array(self, h5py_file_mock,
                                                                                  mock_numpy_save):
        h5py_file_mock.return_value = 'fake mat data'
        go_data = GoData()
        go_data.convert_mat_data_to_numpy_array = Mock()
        go_data.crop_data = Mock()

        go_data.convert_mat_file_to_numpy_file('')

        assert go_data.convert_mat_data_to_numpy_array.call_args_list[1][0] == ('fake mat data', 'depths')
        assert go_data.convert_mat_data_to_numpy_array.call_args_list[0][0] == ('fake mat data', 'images')

    def test_convert_mat_data_to_numpy_array_extracts_and_tranposes_the_data(self):
        go_data = GoData()
        mock_mat_data = Mock()
        mock_mat_data.get.return_value = np.array([[[[1, 2, 3]]]])

        transposed_array = go_data.convert_mat_data_to_numpy_array(mock_mat_data, 'images')

        assert mock_mat_data.get.call_args == (('images',),)
        assert np.array_equal(transposed_array, np.array([[[[1]], [[2]], [[3]]]]))

    @patch('h5py.File')
    @patch('numpy.save')
    def test_convert_mat_file_to_numpy_file_writes_extracted_numpys_to_files(self, mock_numpy_save, h5py_file_mock):
        go_data = GoData()
        go_data.convert_mat_data_to_numpy_array = Mock(side_effect=[1, 2])
        go_data.crop_data = lambda x: x

        go_data.convert_mat_file_to_numpy_file('')

        assert mock_numpy_save.call_args_list[0] == ((os.path.join('images_') + '.npy', 1),)
        assert mock_numpy_save.call_args_list[1] == ((os.path.join('depths_') + '.npy', 2),)

    def test_convert_mat_data_to_numpy_array_can_specify_the_number_of_images_to_extract(self):
        go_data = GoData()
        mock_mat_data = Mock()
        mock_mat_data.get.return_value = np.array([[[[1]]], [[[2]]], [[[3]]]])

        transposed_array = go_data.convert_mat_data_to_numpy_array(mock_mat_data, 'images',
                                                                            number_of_samples=2)

        assert np.array_equal(transposed_array, np.array([[[[1]]], [[[2]]]]))

    @patch('h5py.File')
    @patch('numpy.save')
    def test_convert_mat_file_to_numpy_file_passes_can_be_called_on_a_specific_number_of_images(self,
                                                                                                mock_numpy_save,
                                                                                                h5py_file_mock):
        go_data = GoData()
        mock_convert = Mock()
        go_data.convert_mat_data_to_numpy_array = mock_convert
        go_data.crop_data = Mock()

        go_data.convert_mat_file_to_numpy_file('', number_of_samples=2)

        assert mock_convert.call_args[1]['number_of_samples'] == 2

    def test_crop_data_removes_edge_data(self):
        go_data = GoData()
        array = np.arange(324.0).reshape((1, 18, 18))

        cropped_array = go_data.crop_data(array)

        assert np.array_equal(cropped_array, array[:, 8:10, 8:10])

    def test_rebin_outputs_the_right_types_based_on_dimensions(self):
        four_dimensions = np.array([[[[1]]]])  # Collection of images.
        three_dimensions = np.array([[[1]]])  # Collection of depths.
        go_data = GoData()
        go_data.image_width = 1
        go_data.image_height = 1
        go_data.image_depth = 1

        rebinned_four_dimensions = go_data.shrink_array_with_rebinning(four_dimensions)
        rebinned_three_dimensions = go_data.shrink_array_with_rebinning(three_dimensions)

        assert rebinned_four_dimensions.dtype == np.uint8
        assert rebinned_three_dimensions.dtype == np.float64
