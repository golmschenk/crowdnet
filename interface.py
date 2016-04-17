"""
Interface to help interact with the neural networks.
"""
import multiprocessing


class Interface:
    """
    A class to help interact with the neural networks.
    """
    def __init__(self, network_class):
        self.queue = multiprocessing.Queue()
        self.network = network_class(message_queue=self.queue)

    def train(self):
        """
        Runs the main interactions
        """
        self.network.start()
        while True:
            user_input = input()
            if user_input == 's':
                print('Save requested.')
                self.queue.put('save')
                continue
            elif user_input == 'q':
                print('Exit requested. Quitting.')
                self.queue.put('quit')
                print('Waiting for graph to quit.')
                self.network.join()
                break
