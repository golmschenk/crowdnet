"""
Interface to help interact with the neural networks.
"""
import multiprocessing
import argparse


class Interface:
    """
    A class to help interact with the neural networks.
    """

    def __init__(self, network_class):
        self.queue = multiprocessing.Queue()
        self.network = network_class(message_queue=self.queue)

    def run(self):
        """
        Runs the interface between the user and the network.
        """
        # Setup the parser.
        parser = argparse.ArgumentParser(
            description='Runs the {}. By default, training will be run.'.format(type(self.network).__name__))
        parser.add_argument('-t', '--test',
                            help=('Runs the network on the test data, using the test structure. By default, the model'
                                  'with the latest step number and matching network name will be restored'),
                            action='store_true')
        parser.add_argument('-r', '--restore-model', metavar='MODEL_PATH',
                            help='Used to restore a model. The model path should follow this flag.')
        args = parser.parse_args()

        # Handle any parameters.
        self.network.restore_model = args.restore_model

        # Run the network.
        if args.test:
            self.test()
        else:
            self.train()

        print('Program done.')

    def train(self):
        """
        Runs the main interactions between the user and the network during training.
        """
        self.network.start()
        while True:
            user_input = input()
            if user_input == 's':
                print('Save requested.')
                self.queue.put('save')
            elif user_input == 'q':
                print('Quit requested.')
                self.queue.put('quit')
                self.network.join()
                break
            elif user_input.startswith('l '):
                print('Updating learning rate.')
                self.queue.put('change learning rate')
                self.queue.put(user_input[2:])

    def test(self):
        """
        Runs the network prediction.
        """
        self.network.test()
