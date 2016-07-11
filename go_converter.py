"""
Code for simple conversions between file types.
"""
import os
import shlex
import subprocess

import numpy as np
import re
from PIL import Image


class GoConverter:
    """
    A class for simple conversions of data.
    """

    @staticmethod
    def convert_video_to_images(input_video_path, output_frames_directory, frames_per_second=30):
        """
        Converts a video to images.

        :param input_video_path: The path of the video to convert.
        :type input_video_path: str
        :param output_frames_directory: The path of the directory to ouput the frames. Does not need to exist.
        :type output_frames_directory: str
        :param frames_per_second: The number of images per second of video to generate.
        :type frames_per_second: int
        """
        if not os.path.isdir(output_frames_directory):
            os.mkdir(output_frames_directory)
        output_frames_path = os.path.join(output_frames_directory, r'image_%d.jpg')
        command = 'ffmpeg -i {ivp} -qscale:v 2 -vf fps={fps} {ofp}'.format(ivp=input_video_path,
                                                                           fps=frames_per_second,
                                                                           ofp=output_frames_path)
        print('Running command: %s' % command)
        argument_list = shlex.split(command)
        subprocess.run(argument_list)

    @staticmethod
    def convert_images_to_video(input_frames_directory, output_video_path, frames_per_second=30):
        """
        Converts images to a video.

        :param input_frames_directory: The directory of the images to convert.
        :type input_frames_directory: str
        :param output_video_path: The path to output the video to.
        :type output_video_path: str
        :param frames_per_second: The number of images per second of video.
        :type frames_per_second: int
        """
        input_frames_path = os.path.join(input_frames_directory, r'image_%d.jpg')
        command = 'ffmpeg -i {ifp} -vf fps={fps} {ovp}'.format(ifp=input_frames_path,
                                                               fps=frames_per_second,
                                                               ovp=output_video_path)
        print('Running command: %s' % command)
        argument_list = shlex.split(command)
        subprocess.run(argument_list)

    def convert_images_to_numpy(self, input_images_directory, output_numpy_path):
        """
        Converts a directory of images into a NumPy array.

        :param input_images_directory: The images directory.
        :type input_images_directory: str
        :param output_numpy_path: The path to output the NumPy array to.
        :type output_numpy_path: str
        """
        image_types = ('.jpg', '.jpeg', '.png')
        file_name_list = os.listdir(input_images_directory)
        file_name_list = sorted(file_name_list, key=self.natural_sort_key)
        image_list = []
        for file_name in file_name_list:
            if file_name.endswith(image_types):
                image_file = Image.open(file_name)
                image_file.load()
                image = np.asarray(image_file, dtype="uint8")
                image_list.append(image)
        images = np.stack(image_list)
        np.save(output_numpy_path, images)

    @staticmethod
    def natural_sort_key(sequence, _natural_sort_regex=re.compile('([0-9]+)')):
        """
        A key to allow for natural sorting.
        Taken from: http://stackoverflow.com/a/16090640/1191087

        :param sequence: The sequence to sort.
        :type sequence: list[TypeVar('T')]
        :param _natural_sort_regex: The regex of the natural sort.
        :type _natural_sort_regex: type(re.compile(''))
        :return: The natural sort key.
        :rtype: list[int]
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(_natural_sort_regex, sequence)]

    '''def convert_video_to_numpy(self, input_video_path, output_numpy_path, frames_per_second=30):
        output_directory = os.path.dirname(output_numpy_path)
        temporary_frames_directory = os.path.join(output_directory, 'temporary_frames_directory')
        os.mkdir(temporary_frames_directory)
        self.convert_video_to_images(input_video_path, temporary_frames_directory)

        shutil.rmtree(temporary_frames_directory)'''
