"""
Code for managing the depth data.
"""
from go_data import GoData


class CrowdData(GoData):
    """
    A class for managing the depth data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.height = 158 // 2  # The height we'll be training on (data will be shrunk if needed).
        self.width = 238 // 2  # The width we'll be training on (data will be shrunk if needed).
        self.original_height = 158  # The height of the original data.
        self.original_width = 238  # The width of the original data.


if __name__ == '__main__':
    data = CrowdData(data_directory='data', data_name='crowd',
                     images_numpy_file_name='images.npy', labels_numpy_file_name='densities.npy')
    data.numpy_files_to_tfrecords(augment=True)