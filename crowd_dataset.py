"""
Code for the crowd data dataset.
"""
from collections import namedtuple
import os
import imageio as imageio
import numpy as np
import time
from torch.utils.data import Dataset


CrowdExampleWithPerspective = namedtuple('CrowdExample', ['image', 'label', 'roi', 'perspective'])
CrowdExample = namedtuple('CrowdExample', ['image', 'label', 'roi'])


class CrowdDataset(Dataset):
    """
    A class for the crowd dataset.
    """
    def __init__(self, database_path, data_type, transform=None):
        """
        :param database_path: The path of the dataset directories.
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


class CrowdDatasetWithUnlabeled(Dataset):
    """
    A class for the crowd dataset.
    """
    def __init__(self, database_path, data_type, transform=None):
        """
        :param database_path: The path of the dataset directories.
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
        self.unlabeled_video_reader = imageio.get_reader(os.path.join(dataset_directory, 'unlabeled_images.avi'))
        unlabeled_image_counts = np.load(os.path.join(dataset_directory, 'unlabeled_image_counts.npy'), mmap_mode='r')
        unlabeled_lookup = [unlabeled_image_counts[0]]
        for unlabeled_image_count in unlabeled_image_counts[1:]:
            unlabeled_lookup.append(unlabeled_lookup[-1] + unlabeled_image_count)
        self.unlabeled_lookup = np.array(unlabeled_lookup, dtype=np.int32)
        self.unlabeled_rois = np.load(os.path.join(dataset_directory, 'unlabeled_rois.npy'), mmap_mode='r')
        self.unlabeled_perspectives = np.load(os.path.join(dataset_directory, 'unlabeled_perspectives.npy'),
                                              mmap_mode='r')
        self.length = self.labels.shape[0]
        self.unlabeled_length = self.unlabeled_lookup[-1]
        self.transform = transform

    def __getitem__(self, index):
        """
        :param index: The index within the entire dataset.
        :type index: int
        :return: An example from the crowd dataset.
        :rtype: CrowdExample, CrowdExample
        """
        example = CrowdExampleWithPerspective(image=self.images[index], label=self.labels[index], roi=self.rois[index],
                                              perspective=self.perspectives[index])
        if self.transform:
            example = self.transform(example)

        unlabeled_example = None
        for attempt in range(3):  # Azure hard drive acts up once and a while during this read. Try again.
            try:
                unlabeled_index = np.random.randint(self.unlabeled_length)
                unlabeled_camera_index = np.searchsorted(self.unlabeled_lookup, unlabeled_index)
                unlabeled_image = self.unlabeled_video_reader.get_data(unlabeled_camera_index)
                unlabeled_example = CrowdExampleWithPerspective(image=unlabeled_image,
                                                                label=np.zeros(shape=(unlabeled_image.shape[:2]),
                                                                               dtype=np.float32),
                                                                roi=self.unlabeled_rois[unlabeled_camera_index],
                                                                perspective=self.unlabeled_perspectives[
                                                                    unlabeled_camera_index])
            except imageio.core.format.CannotReadFrameError as error:
                if attempt == 2:
                    raise error
                time.sleep(60)
                continue
            break

        if self.transform:
            unlabeled_example = self.transform(unlabeled_example)

        return example, unlabeled_example

    def __len__(self):
        return self.length
