"""
Code for the settings of the network.
"""
from gonet.settings import Settings as GoSettings


class Settings(GoSettings):
    """
    A class for the settings of the network.
    """
    def __init__(self):
        super().__init__()

        self.network_name = '1 Camera GAN'
        self.inference_op_name = 'experimental'
        self.datasets_json = '../storage/data/1_camera.json'

        self.batch_size = 20
        self.initial_learning_rate = 0.001
        self.learning_rate_decay_rate = 0.1
        self.learning_rate_decay_steps = 30000

        self.image_height = 576 // 8
        self.image_width = 720 // 8
        self.image_depth = 3
