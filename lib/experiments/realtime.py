from __future__ import absolute_import, print_function

import cv2
import time
import numpy as np

from . import Experiment
from ..metrics import iou
from ..utils.viz import show_frame


class ExperimentRealtime(Experiment):
    
    def __init__(self, log_dir=None, **kargs):
        super(ExperimentRealtime, self).__init__(
            experiment_type='realtime', log_dir=log_dir)
        self.parse_args(**kargs)

    def parse_args(self, **kargs):
        default_args = {
            'repetitions': 1,
            'default_fps': 20,
            'critical': True,
            'grace': 3,
            'burnin': 10,
            'override_fps': True,
            'skip_initialize': 5,
            'skip_tags': [],
            'failure_overlap': 0,
            'realtime_type': 'real'}
        
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
                if r == 3 and self.check_deterministic(self.log_dir, seq_name):
                    print('detected a deterministic tracker, ' +
                          'skipping remaining trails.')
                    break
                
                print('  repetition:', r + 1)
                states = []
                times = []

                init_frame = 0

                # tracking loop
                for f, img_file in enumerate(img_files):
                    image = cv2.imread(img_file)
                    if image.ndim == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.ndim == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    start_time = time.time()
                    if f == init_frame:
                        tracker.init(image, anno[f])
                        elapsed_time = time.time() - start_time
                        states.append([1])

                        acc_time = 1. / self.default_fps
                        grace = self.grace - 1
                        failed = False
                    elif failed:
                        states.append([0])
                        elapsed_time = np.nan
                    else:
                        if grace > 0:
                            state = tracker.update(image)
                            elapsed_time = time.time() - start_time
                            acc_time += 1. / self.default_fps
                            grace -= 1
                            if grace == 0:
                                next_frame = init_frame + round(np.floor((acc_time + max(1. / self.default_fps, elapsed_time)) * self.default_fps))
                        elif f < next_frame:
                            state = state
                            elapsed_time = np.nan
                        else:
                            state = tracker.update(image)
                            elapsed_time = time.time() - start_time
                            acc_time += max(1. / self.default_fps, elapsed_time)
                            next_frame = init_frame + round(np.floor((acc_time + max(1. / self.default_fps, elapsed_time)) * self.default_fps))
                        
                        if iou(state, anno[f]) > self.failure_overlap:
                            states.append(state)
                        else:
                            failed = True
                            states.append([2])
                            init_frame = next_frame + self.skip_initialize
                    times.append(elapsed_time)
                    
                    if visualize:
                        if f == init_frame:
                            show_frame(image, anno[f],
                                       color=(0, 0, 255), pause=0.5)
                        else:
                            show_frame(image, state,
                                       color=(255, 0, 0), pause=0.001)
            
                # record tracking results
                self.record(r, seq_name, states, times)
