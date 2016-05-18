"""
Code to help import TFEvents data.
"""
from tensorflow.python.summary.event_accumulator import EventAccumulator, ScalarEvent


class TFEventsImporter:
    """
    A class to import TFEvents data.
    """

    def __init__(self, tfevents_path):
        """
        :param tfevents_path: The path to the TFEvents file.
        :type tfevents_path: str
        """
        self.tfevents_path = tfevents_path
        self.event_accumulator = EventAccumulator(tfevents_path)
        self.event_accumulator.Reload()

    def get_scalar_events_for_name(self, scalar_name):
        """
        Retrieves the ScalarEvents for a given scalar name.

        :param scalar_name: The name of the scalar.
        :type scalar_name: str
        :return: list[ScalarEvent]
        :rtype:
        """
        scalar_events = self.event_accumulator._scalars._buckets[scalar_name].items
        assert len(scalar_events) != 0
        return scalar_events
