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

        self.network_name = 'crowd_net'
        self.inference_op_name = 'bn_gaea'
        self.datasets_json = 'datasets.json'

        self.batch_size = 3
        self.initial_learning_rate = 0.001
        self.learning_rate_decay_rate = 0.1
        self.learning_rate_decay_steps = 20000

        self.image_height = 240 // 4
        self.image_width = 352 // 4
        self.image_depth = 3
