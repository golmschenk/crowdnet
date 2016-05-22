"""
Tests for the TFEventsImporter
"""

import os

from tfevents_importer import TFEventsImporter


class TestFunctionalTFEventsImporter:
    """
    A test suite for the TFEventsImporter class.
    """
    def test_can_import_scalar_data(self):
        tfevents_importer = TFEventsImporter(os.path.join('functional_tests', 'test_data', 'small.tfevents'))
        scalar_events = tfevents_importer.get_scalar_events_for_name('Loss per pixel')

        assert len(scalar_events) == 46
        assert scalar_events[2].value == 0.9873210787773132
