"""
Tests for the convenience module.
"""
from convenience import size_from_stride_two


class TestConvenience:
    def test_size_from_stride_two_gives_correct_size(self):
        size = size_from_stride_two(49)

        assert size == 25

    def test_size_from_stride_two_gives_correct_size_for_multiple_iterations(self):
        size = size_from_stride_two(49, iterations=2)

        assert size == 13
