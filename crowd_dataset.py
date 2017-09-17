"""
Code for the crowd data dataset.
"""
from collections import namedtuple
import os
import numpy as np
from torch.utils.data import Dataset


CrowdExampleWithPerspective = namedtuple('CrowdExample', ['image', 'label', 'roi', 'perspective'])
CrowdExample = namedtuple('CrowdExample', ['image', 'label', 'roi'])


class CrowdDataset(Dataset):
    """
    A class for the crowd dataset.
    """

    def __init__(self, database_path, data_type, transform=None):
        """
        :param database_path: The path of the HDF5 database file.
        :type database_path: str
        :param data_type: The type of data to be loaded (e.g. train, test, etc).
        :type data_type: str
        :param transform: The transformations to be applied to the dataset.
        :type transform: callable
        """
        dataset_directory = os.path.join(database_path, data_type)
        self.images = np.load(os.path.join(dataset_directory, 'images.npy'), mmap_mode='r')
        self.labels = np.load(os.path.join(dataset_directory, 'labels.npy'), mmap_mode='r')
        self.rois = np.load(os.path.join(dataset_directory, 'rois.npy'), mmap_mode='r')
        self.perspectives = np.load(os.path.join(dataset_directory, 'perspectives.npy'), mmap_mode='r')
        self.length = self.labels.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example from the crowd dataset.
        :rtype: CrowdExample
        """
        example = CrowdExampleWithPerspective(image=self.images[index], label=self.labels[index], roi=self.rois[index],
                                              perspective=self.perspectives[index])
        if self.transform:
            example = self.transform(example)
        return example

    def __len__(self):
        return self.length
