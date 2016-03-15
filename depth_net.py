"""
Code related to the DepthNet.
"""

import tensorflow as tf
import time

from data import Data


class DepthNet:
    """
    The class to build and interact with the DepthNet TensorFlow graph.
    """

    def __int__(self):
        self.batch_size = 32
        self.number_of_epochs = 10000
        self.initial_learning_rate = 0.01

    def train_network(self):
        with tf.Graph().as_default():
            # Setup the inputs.
            data = Data(data_directory='examples', data_name='nyud_micro')
            images, depths = data.inputs(data_type='train', batch_size=self.batch_size,
                                         num_epochs=self.number_of_epochs)

            # Add the forward pass operations to the graph.
            predicted_depths = self.inference(images)

            # Add the loss operations to the graph.
            loss = self.loss(predicted_depths, depths)

            # Add the training operations to the graph.
            train_op = self.training(loss, self.initial_learning_rate)

            # The op for initializing the variables.
            initialize_op = tf.initialize_all_variables()

            # Create a session for running operations in the Graph.
            session = tf.Session()

            # Initialize the variables.
            session.run(initialize_op)

            # Start input enqueue threads.
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

            step = 0
            try:
                while not coordinator.should_stop():
                    start_time = time.time()

                    # Run one step of the model.  The return values are
                    # the activations from the `train_op` (which is
                    # discarded) and the `loss` op.  To inspect the values
                    # of your ops or variables, you may include them in
                    # the list passed to sess.run() and the value tensors
                    # will be returned in the tuple from the call.
                    _, loss_value = session.run([train_op, loss])

                    duration = time.time() - start_time

                    # Print an overview fairly often.
                    if step % 100 == 0:
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                                   duration))
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (self.number_of_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coordinator.request_stop()

            # Wait for threads to finish.
            coordinator.join(threads)
            session.close()
