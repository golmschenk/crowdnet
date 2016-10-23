"""
Code for the settings of the network.
"""
from gonet.settings import Settings as GoSettings


class Settings(GoSettings):
    """
    A class for the settings of the network.
    """
    def __init__(self):
        self.network_name = 'crowd_net'

        self.batch_size = 3
        self.initial_learning_rate = 0.00001
        self.learning_rate_decay_rate = 0.96
        self.learning_rate_decay_steps = 10000
