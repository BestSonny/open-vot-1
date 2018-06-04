from __future__ import absolute_import

import os
import glob
import numpy as np


class Experiment(object):

    def __init__(self, experiment_type, log_dir=None):
        self.experiment_type = experiment_type
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = 'results/anonymous'
        self.log_dir = os.path.join(self.log_dir, experiment_type)

    def run(self, tracker, dataset, visualize=False):
        raise NotImplementedError()

    def check_deterministic(self, log_dir, seq_name):
        log_dir = os.path.join(self.log_dir, seq_name)
        states_files = sorted(glob.glob(os.path.join(
            log_dir, '%s_[0-9]*.txt' % seq_name)))
        if len(states_files) < 3:
            return False

        states_all = []
        for states_file in states_files:
            with open(states_file, 'r') as f:
                states_all.append(f.read())

        return len(set(states_all)) == 1

    def record(self, repeat, seq_name, states, times):
        log_dir = os.path.join(self.log_dir, seq_name)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        states_file = os.path.join(
            log_dir, '%s_%03d.txt' % (seq_name, repeat + 1))
        times_file = os.path.join(log_dir, '%s_time.txt' % seq_name)

        # record tracking results
        states_str = []
        for state in states:
            assert len(state) in [1, 4, 6]
            if len(state) == 1:
                states_str.append('%d' % state[0])
            else:
                states_str.append(str.join(',', ['%.4f' % s for s in state]))
        states_str = str.join('\n', states_str)
        with open(states_file, 'w') as f:
            f.write(states_str)

        # record tracking times
        if not os.path.isfile(times_file):
            times_arr = np.zeros((len(times), self.repetitions))
            np.savetxt(times_file, times_arr, fmt='%.6f', delimiter=',')
        else:
            times_arr = np.loadtxt(times_file, delimiter=',')
        if times_arr.ndim == 1:
            times_arr = times_arr[:, np.newaxis]
        times_arr[:, repeat] = times
        np.savetxt(times_file, times_arr, fmt='%.6f', delimiter=',')


from .unsupervised import ExperimentUnsupervised
from .supervised import ExperimentSupervised
