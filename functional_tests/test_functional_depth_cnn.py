"""
Functional tests for the depth CNN.
"""
from depth_net import DepthNet
import tensorflow as tf


class TestFunctionalDepthCnn:
    def test_basic_cnn_can_produce_loss(self):
        session = tf.Session()
        depth_net = DepthNet()

        loss = session.run(depth_net.loss)

        assert loss > 1  # The loss should certainly be greater than 1 without training (probably even with it)
