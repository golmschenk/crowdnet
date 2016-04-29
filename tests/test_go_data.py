"""
Tests for code related to GoData.
"""
from unittest.mock import patch
import numpy as np

from go_data import GoData


class TestGoData:
    """
    A class for the GoData test suite.
    """
    @patch('numpy.random.permutation')
    def test_data_shuffling(self, mock_permutation):
        go_data = GoData()
        go_data.images = np.array([1, 2, 3])
        go_data.labels = np.array(['a', 'b', 'c'])
        mock_permutation.return_value = [2, 0, 1]

        go_data.shuffle()

        go_data.images = np.array([3, 1, 2])
        go_data.labels = np.array(['c', 'a', 'b'])
