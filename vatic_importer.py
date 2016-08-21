"""
Code for importing data from Vatic.
"""
from subprocess import call
import os


class VaticImporter:
    """
    A class for working with data from Vatic.
    """

    def __init__(self, identifier_name, frames_directory, vatic_directory, output_directory):
        self.identifier_name = identifier_name
        self.frames_directory = os.path.abspath(frames_directory)
        self.vatic_directory = os.path.abspath(vatic_directory)
        self.output_directory = os.path.abspath(output_directory)

    def dump_vatic_data_to_text(self):
        """
        Dumps the Vatic data for the video to a text file.
        """
        output_file = os.path.join(self.output_directory, 'text_dump.txt')
        call('(cd {} && exec turkic dump {} -o {})'.format(self.vatic_directory, self.identifier_name,
                                                           output_file).split(' '))
