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
        parser = argparse.ArgumentParser(description='Runs the {}.'.format(type(self.network).__name__))
        parser.add_argument('--test', help='Runs the network on the test data, using the test structure',
                            action='store_true')
        args = parser.parse_args()

        if args.test:
            self.predict()
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

    def predict(self):
        """
        Runs the network prediction.
        """
        self.network.predict()
