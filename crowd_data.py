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
        self.image_height = 480 // 8  # The height we'll be training on (data will be resized if needed).
        self.image_width = 704 // 8  # The width we'll be training on (data will be resized if needed).
        self.train_size = 'all'
        self.dataset_type = None

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
                    if self.dataset_type == 'deploy':
                        images_path = os.path.abspath(os.path.join(file_directory, numpy_file_name))
                        import_file_paths.append(images_path)
                        continue
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
        if self.dataset_type == 'deploy':
            image_file_path = import_file_path
        else:
            image_file_path = import_file_path[0]
        export_name = os.path.splitext(os.path.basename(image_file_path))[0]
        self.data_name = export_name.replace('images', 'data_pairs').replace('image', 'data_pair')

    def import_file(self, file_path):
        """
        Import the data.
        Should be overwritten by subclasses.

        :param file_path: The file path of the file to be imported.
        :type file_path: str | (str, str)
        """
        if self.dataset_type == 'deploy':
            self.import_numpy_images_file(file_path)
        else:
            self.import_numpy_pair_files(file_path)

    def import_numpy_images_file(self, file_path):
        """
        Imports the images from the file path.

        :param file_path: The file path of the numpy image.
        :type file_path: str
        """
        images = np.load(file_path)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        self.images = images
        self.labels = None

    def import_numpy_pair_files(self, file_path_pair):
        """
        Imports the images and labels from the file path pair.

        :param file_path_pair: The pair of file paths.
        :type file_path_pair: (str, str)
        """
        images = np.load(file_path_pair[0])
        labels = np.load(file_path_pair[1]).astype(np.float32)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
            labels = np.expand_dims(labels, axis=0)
        self.images = images
        self.labels = labels


if __name__ == '__main__':
    data = CrowdData()
    data.generate_all_tfrecords()
