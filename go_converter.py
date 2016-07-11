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
