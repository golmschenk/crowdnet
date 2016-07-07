"""
Code for simple conversions between file types.
"""
import os
import shlex
import subprocess


class GoConverter:
    """
    A class for simple conversions of data.
    """
    def convert_video_to_images(self, input_video_path, output_frames_directory, frames_per_second=30):
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
        command = 'ffmpeg -i {ivp} -vf fps={fps} {ofp}'.format(ivp=input_video_path,
                                                               fps=frames_per_second,
                                                               ofp=output_frames_path)
        print('Running command: %s' % command)
        argument_list = shlex.split(command)
        subprocess.run(argument_list)
