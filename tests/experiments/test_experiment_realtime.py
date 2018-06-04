from __future__ import absolute_import

import unittest

from lib.experiments import ExperimentRealtime
from lib.trackers import TrackerSiamFC
from lib.datasets import VOT


class TestExperimentRealtime(unittest.TestCase):

    def setUp(self):
        self.vot_dir = 'data/vot2017'
        self.net_path = 'pretrained/siamfc/2016-08-17.net.mat'

    def tearDown(self):
        pass

    def test_experiment_realtime(self):
        experiment = ExperimentRealtime(log_dir='results/SiamFC')
        tracker = TrackerSiamFC(net_path=self.net_path)
        dataset = VOT(self.vot_dir, return_rect=True)
        experiment.run(tracker, dataset, visualize=True)


if __name__ == '__main__':
    unittest.main()
