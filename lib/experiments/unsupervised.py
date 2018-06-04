from __future__ import absolute_import, print_function

import cv2
import time

from . import Experiment
from ..utils.viz import show_frame


class ExperimentUnsupervised(Experiment):

    def __init__(self, log_dir=None, **kargs):
        super(ExperimentUnsupervised, self).__init__(
            experiment_type='unsupervised', log_dir=log_dir)
        self.parse_args(**kargs)

    def parse_args(self, **kargs):
        default_args = {
            'repetitions': 1,
            'burnin': 0}

        for key, val in default_args.items():
            if key in kargs:
                setattr(self, key, kargs[key])
            else:
                setattr(self, key, val)

    def run(self, tracker, dataset, visualize=False):
        for s, (img_files, anno) in enumerate(dataset):
            seq_name = dataset.seq_names[s]
            print('sequence:', seq_name)

            for r in range(self.repetitions):
                if r == 4 and self.check_deterministic(self.log_dir, seq_name):
                    print('detected a deterministic tracker,' +
                          'skippning remaining trails.')
                    break

                print('  repetition:', r + 1)
                states = []
                times = []

                # tracking loop
                for f, img_file in enumerate(img_files):
                    image = cv2.imread(img_file)
                    if image.ndim == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.ndim == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    start_time = time.time()
                    if f == 0:
                        tracker.init(image, anno[f])
                        states.append([1])
                    else:
                        state = tracker.update(image)
                        states.append(state)
                    times.append(time.time() - start_time)

                    if visualize:
                        show_frame(image, state if f > 0 else anno[f])

                # record tracking results
                self.record(r, seq_name, states, times)
