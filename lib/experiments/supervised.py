from __future__ import absolute_import, print_function

import cv2
import time
import numpy as np

from . import Experiment
from ..metrics import iou
from ..utils.viz import show_frame


class ExperimentSupervised(Experiment):
    
    def __init__(self, log_dir=None, **kargs):
        super(ExperimentSupervised, self).__init__(
            experiment_type='supervised', log_dir=log_dir)
        self.parse_args(**kargs)

    def parse_args(self, **kargs):
        default_args = {
            'repetitions': 15,
            'burnin': 10,
            'skip_tags': [],
            'skip_initialize': 5,
            'failure_overlap': 0}
        
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
                failed = False
                passed_frames = -1

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
                        states.append([1])  # "1" indicates the initialization frame
                    elif not failed:
                        state = tracker.update(image)
                        if iou(state, anno[f]) > self.failure_overlap:
                            states.append(state)
                        else:
                            failed = True
                            passed_frames = 1
                            states.append([2])  # "2" indicates the failed frame
                    else:
                        if passed_frames < self.skip_initialize:
                            passed_frames += 1
                            states.append([0])  # "0" indicates the skipped frame
                            start_time = np.nan
                        else:
                            tracker.init(image, anno[f])
                            failed = False
                            passed_frames = -1
                            states.append([1])
                    times.append(time.time() - start_time)

                    if visualize:
                        is_init = len(states[-1]) == 1 and states[-1][0] == 1
                        if is_init:
                            show_frame(image, anno[f],
                                       color=(0, 0, 255), pause=0.5)
                        else:
                            show_frame(image, state,
                                       color=(255, 0, 0), pause=0.001)
                
                # record tracking results
                self.record(r, seq_name, states, times)
