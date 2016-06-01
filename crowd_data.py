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
        self.height = 158 // 2  # The height we'll be training on (data will be shrunk if needed).
        self.width = 238 // 2  # The width we'll be training on (data will be shrunk if needed).
        self.original_height = 158  # The height of the original data.
        self.original_width = 238  # The width of the original data.
        self.image_shape = [self.height, self.width, self.channels]
        self.label_shape = [self.height, self.width, 1]

    def augment_data_set(self):
        """
        Augments the data set with some basic approaches.
        """
        self.offset_augmentation(1)
        self.gaussian_noise_augmentation(12, 8)


if __name__ == '__main__':
    data = CrowdData()
    data.numpy_files_to_tfrecords(augment=True)
