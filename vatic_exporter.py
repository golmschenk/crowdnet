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

    def __init__(self, identifier, vatic_directory, output_root_directory, frames_root_directory=None):
        self.identifier = identifier
        if frames_root_directory:
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

    def extract_head_y_and_height_array_from_text_dump(self):
        """
        Gets head y positions paired with heights from a text dump file containing head and person labels.

        :return: The array containing the head y and height values.
        :rtype: np.ndarray
        """
        frame_objects_dict = {}
        with open(self.text_dump_filename) as text_dump_file:
            text_dump_content = csv.reader(text_dump_file, delimiter=' ')
            for row in text_dump_content:
                out_of_frame = row[6]
                if out_of_frame == '1':
                    continue
                if 'Head' not in row and 'Person' not in row:
                    continue
                frame_number = row[5]
                if frame_number not in frame_objects_dict:
                    frame_objects_dict[frame_number] = {'Heads': [], 'People': []}
                if 'Head' in row:
                    head = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
                    frame_objects_dict[frame_number]['Heads'].append(head)
                if 'Person' in row:
                    person = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
                    frame_objects_dict[frame_number]['People'].append(person)

        # Find matching head people pairs
        pairs = []
        for frame_number, frame_objects in frame_objects_dict.items():
            heads = frame_objects['Heads']
            people = frame_objects['People']
            for head in heads:
                for person in people:
                    # Matching check.
                    if person[0] < head[0] < head[2] < person[2] and abs(head[1] - person[1]) < 5:
                        if (head, person) not in pairs:
                            # Only allow unique combinations to prevent people who just stand still for a long time
                            # from being over weighted for.
                            pairs.append((head, person))
                        break  # One head per person.

        # Keep only the head y position and the height.
        head_y_and_height_list = []
        for pair in pairs:
            head = pair[0]
            person = pair[1]
            y = int(head[1] + head[3]) // 2
            height = person[3] - person[1]
            head_y_and_height_list.append((y, height))
        return np.stack(head_y_and_height_list)

    def calculate_height_to_head_position_polynomial_fit_from_text_dump(self, polynomial_degree=1):
        """
        Calculates the coefficients of the polynomial fit from the head position to the height.

        :param polynomial_degree: The degree of polynomial to fit to the data.
        :type polynomial_degree: int
        :return: The array containing the coefficients. Highest degree first.
        :rtype: np.ndarray
        """
        head_y_and_height_array = self.extract_head_y_and_height_array_from_text_dump()
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
        # Subparsers.
        subparsers = parser.add_subparsers()

        # Generic arguments.
        generic_parser = argparse.ArgumentParser(add_help=False)
        generic_parser.add_argument('--identifier', type=str,
                                    help=('The identifier of the video in Vatic (should alsobe the name of the'
                                          'subdirectory in frames root directory).'))
        generic_parser.add_argument('--vatic_directory', type=str,
                                    help='The vatic directory to run Turkic from.')
        generic_parser.add_argument('--output_root_directory', type=str, help='The path to export the data to.')

        # Head subparser specific arguments.
        head_export_parser = subparsers.add_parser('head', help='Exports the head positions.', parents=[generic_parser])
        head_export_parser.add_argument('--frames_root_directory', type=str,
                                        help='The parent directory in which all Vatic video frames are stored')
        head_export_parser.set_defaults(program='head')

        # Height subparser specific arguments.
        height_export_parser = subparsers.add_parser('height', help='Exports the coefficients for the height fitting.',
                                                     parents=[generic_parser])
        height_export_parser.set_defaults(program='height')

        args = parser.parse_args()

        # Create the exporter.
        if not hasattr(args, 'frames_root_directory'):
            args.frames_root_directory = None
        vatic_exporter = cls(args.identifier, args.vatic_directory, args.output_root_directory,
                             args.frames_root_directory)

        vatic_exporter.dump_vatic_data_to_text()
        if args.program == 'head':
            vatic_exporter.create_head_point_position_files_from_text_dump()
        elif args.program == 'height':
            print(vatic_exporter.calculate_height_to_head_position_polynomial_fit_from_text_dump())


if __name__ == '__main__':
    VaticExporter.command_line_interface()