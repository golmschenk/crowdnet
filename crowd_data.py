"""
Code for managing the crowd data.
"""
import numpy as np
import os

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

    def attain_import_file_paths(self):
        """
        Gets a list of all the file paths for files to be imported.

        :return: The list of the file paths to be imported.
        :rtype: list[str]
        """
        import_file_paths = []
        for file_directory, _, file_names in os.walk(self.import_directory):
            numpy_file_names = [file_name for file_name in file_names if file_name.endswith('.npy')]
            for numpy_file_name in numpy_file_names:
                if 'image' in numpy_file_name:
                    images_path = os.path.abspath(os.path.join(file_directory, numpy_file_name))
                    if os.path.isfile(images_path.replace('image', 'density')):
                        labels_path = images_path.replace('image', 'density')
                    elif os.path.isfile(images_path.replace('images', 'densities')):
                        labels_path = images_path.replace('images', 'densities')
                    elif os.path.isfile(images_path.replace('image', 'label')):
                        labels_path = images_path.replace('image', 'label')
                    elif os.path.isfile(images_path.replace('images', 'labels')):
                        labels_path = images_path.replace('images', 'labels')
                    else:
                        continue
                    import_file_paths.append((images_path, labels_path))
        return import_file_paths

    def obtain_export_name(self, import_file_path):
        """
        Extracts the name to be used for the export file.

        :param import_file_path: The import path.
        :type import_file_path: str | (str, str)
        :return: The name of the export file.
        :rtype: str
        """
        image_file_path = import_file_path[0]
        export_name = os.path.splitext(os.path.basename(image_file_path))[0]
        self.data_name = export_name.replace('images', 'data_pair').replace('image', 'data_pair')

    def import_file(self, file_path):
        """
        Import the data.
        Should be overwritten by subclasses.

        :param file_path: The file path of the file to be imported.
        :type file_path: str | (str, str)
        """
        self.import_numpy_pair_files(file_path)

    def import_numpy_pair_files(self, file_path_pair):
        """
        Imports the images and labels from the file path pair.

        :param file_path_pair: The pair of file paths.
        :type file_path_pair: (str, str)
        """
        self.images = np.load(file_path_pair[0])
        self.labels = np.load(file_path_pair[1])

if __name__ == '__main__':
    data = CrowdData()
    data.generate_all_tfrecords()
