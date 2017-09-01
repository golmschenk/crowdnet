"""
Code for the crowd data dataset.
"""
import json
from collections import namedtuple

import os

import numpy as np
from torch.utils.data import Dataset


CrowdDatasetEntry = namedtuple('CrowdDatasetEntry', ['file_name', 'example_count', 'start_index'])
CrowdExample = namedtuple('CrowdExample', ['image', 'label', 'roi'])


class CrowdDataset(Dataset):
    """
    A class for the crowd dataset.

    :type entries: list[CrowdDatasetEntry]
    """

    def __init__(self, root_directory, dataset_json_file_name, data_type, transform=None):
        """
        :param dataset_json_file_name: The file name of the json file containing the dataset file listings.
        :type dataset_json_file_name: str
        :param root_directory: The path to the root directory of the dataset.
        :type root_directory: str
        :param data_type: The type of data to be loaded (e.g. train, test, etc).
        :type data_type: str
        :param transform: The transformations to be applied to the dataset.
        :type transform: callable
        """
        self.root_directory = root_directory
        self.entries = []
        with open(os.path.join(root_directory, dataset_json_file_name)) as dataset_json_file:
            json_dict = json.load(dataset_json_file)
        total_count = 0
        for file_name in json_dict[data_type]:
            example_array = np.load(os.path.join(root_directory, file_name + '_images.npy'), mmap_mode='r')
            example_count = example_array.shape[0]
            start_index = total_count
            self.entries.append(CrowdDatasetEntry(file_name, example_count, start_index))
            total_count += example_count
        self.total_example_count = total_count
        self.transform = transform

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example from the crowd dataset.
        :rtype: CrowdExample
        """
        example = None
        for entry in self.entries:
            if index < entry.start_index + entry.example_count:
                example = self.load_example(entry.file_name, index - entry.start_index)
                break
        try:
            assert example is not None
        except AssertionError:
            raise IndexError('Index {} out of range for dataset {}.'.format(index, self))
        if self.transform:
            example = self.transform(example)
        return example

    def __len__(self):
        return self.total_example_count

    def load_example(self, file_name, index):
        """
        Loads an example from it's respective files.

        :param file_name: The base file name for the data.
        :type file_name: str
        :param index: The index of the example within the base file.
        :type index: int
        """
        image = np.load(os.path.join(self.root_directory, file_name + '_images.npy'), mmap_mode='r')[index]
        label = np.load(os.path.join(self.root_directory, file_name + '_labels.npy'), mmap_mode='r')[index]
        roi = np.load(os.path.join(self.root_directory, file_name + '_roi.npy'), mmap_mode='r')
        return CrowdExample(image=image, label=label, roi=roi)
