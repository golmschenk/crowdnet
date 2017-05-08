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
        self.inference_op_name = 'experimental'
        self.datasets_json = 'datasets.json'

        self.batch_size = 25
        self.initial_learning_rate = 0.001
        self.learning_rate_decay_rate = 0.1
        self.learning_rate_decay_steps = 20000

        self.image_height = 576 // 6
        self.image_width = 720 // 6
        self.image_depth = 3
