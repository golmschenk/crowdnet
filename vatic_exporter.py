"""
Code for exporting data from Vatic.
"""
import csv
import os
from shutil import copyfile
from subprocess import call
import numpy as np
import argparse


class VaticExporter:
    """
    A class for working with data from Vatic.
    """

    def __init__(self, identifier, frames_root_directory, vatic_directory, output_root_directory):
        self.identifier = identifier
        self.frames_directory = os.path.join(os.path.abspath(frames_root_directory), self.identifier)
        self.vatic_directory = os.path.abspath(vatic_directory)
        self.output_directory = os.path.join(os.path.abspath(output_root_directory), self.identifier)
        if not os.path.isdir(self.output_directory):
            os.mkdir(self.output_directory)
        self.text_dump_filename = os.path.join(self.output_directory, 'text_dump.txt')

    def dump_vatic_data_to_text(self):
        """
        Dumps the Vatic data for the video to a text file.
        """
        call('/usr/local/bin/turkic dump {} -o {}'.format(self.identifier, self.text_dump_filename).split(' '),
             cwd=self.vatic_directory)

    def create_head_point_position_files_from_text_dump(self, copy_frame_image=True):
        """
        Creates the head position Numpy files from the text dump.
        """
        with open(self.text_dump_filename) as text_dump_file:
            text_dump_content = csv.reader(text_dump_file, delimiter=' ')
            for row in text_dump_content:
                out_of_frame = row[6]
                if out_of_frame == '1' or 'Head' not in row:
                    continue
                frame_number = row[5]
                x0, x1 = int(row[1]), int(row[3])
                y0, y1 = int(row[2]), int(row[4])
                x = (x0 + x1) // 2
                y = (y0 + y1) // 2
                frame_file_path = self.get_frame_file_path(frame_number)
                frame_filename = os.path.basename(frame_file_path)
                frame_filename_without_extension = os.path.splitext(frame_filename)[0]
                numpy_path = os.path.join(self.output_directory, frame_filename_without_extension + '.npy')
                if os.path.isfile(numpy_path):
                    head_position_array = np.load(numpy_path)
                    new_head_position = np.array([[x, y]])
                    head_position_array = np.concatenate((head_position_array, new_head_position))
                    np.save(numpy_path, head_position_array)
                    if copy_frame_image:
                        copyfile(frame_file_path, os.path.join(self.output_directory, frame_filename))
                else:
                    np.save(numpy_path, np.array([[x, y]]))

    def calculate_polynomial_fit_from_text_dump(self, polynomial_degree):
        head_y_and_height_array = self.extract_head_y_and_height_array()
        head_y_array = head_y_and_height_array[:, 0]
        height_array = head_y_and_height_array[:, 1]
        coefficients = np.polyfit(head_y_array, height_array, polynomial_degree)
        return coefficients

    def get_frame_file_path(self, frame_number):
        """
        Finds the path to the frame in the vatic frame directory.

        :param frame_number: The number of the frame whose path is to be retrieved.
        :type frame_number: int
        :return: The full path to the frame.
        :rtype: str
        """
        for root, directories, filenames in os.walk(self.frames_directory):
            for filename in filenames:
                if filename == '{}.jpg'.format(frame_number):
                    return os.path.join(root, filename)

    @classmethod
    def command_line_interface(cls):
        """
        Allows for running the exporter from the command line.
        """
        # Parse arguments.
        parser = argparse.ArgumentParser(
            description='Exports data from Vatic into other forms. The default is exporting to head positions.'
        )
        parser.add_argument('vatic_directory', help='The vatic directory to run Turkic from.')
        parser.add_argument('frames_root_directory',
                            help='The parent directory in which all Vatic video frames are stored')
        parser.add_argument('identifier', help=('The identifier of the video in Vatic (should also be the name of the'
                                                'subdirectory in frames root directory).'))
        parser.add_argument('output_root_directory', help='The path to export the data to.')
        args = parser.parse_args()

        # Create the exporter.
        vatic_exporter = cls(args.identifier, args.frames_root_directory, args.vatic_directory,
                             args.output_root_directory)

        # Run the exporter.
        vatic_exporter.dump_vatic_data_to_text()
        vatic_exporter.create_head_point_position_files_from_text_dump()


if __name__ == '__main__':
    VaticExporter.command_line_interface()
