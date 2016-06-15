"""
Code for managing the crowd data.
"""
from go_data import GoData


class CrowdData(GoData):
    """
    A class for managing the crowd data.
    """
    def __init__(self):
        super().__init__()

        self.data_directory = 'data'
        self.data_name = 'human'
        self.images_numpy_file_name = 'images.npy'
        self.labels_numpy_file_name = 'densities.npy'
        self.dataset_container = 'file'
        self.image_height = 158 // 2  # The height we'll be training on (data will be shrunk if needed).
        self.image_width = 238 // 2  # The width we'll be training on (data will be shrunk if needed).
        self.train_size = 'all'


if __name__ == '__main__':
    data = CrowdData()
    data.numpy_files_to_tfrecords()
