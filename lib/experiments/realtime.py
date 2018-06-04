from __future__ import absolute_import, print_function

import cv2
import time
import math

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
                        states.append([1])

                        failed = False
                        grace = self.grace - 1
                    elif not failed:
                        if grace > 0:
                            state = tracker.update(image)

                            grace -= 1
                            if grace == 1:
                                elapsed_time = time.time() - start_time
                                next_frame = f + max(1, round(math.floor(elapsed_time * self.default_fps)))
                        elif f == next_frame:
                            state = tracker.update(image)
                        else:
                            state = state
                        
                        if iou(state, anno[f]) > self.failure_overlap:
                            states.append(state)
                        else:
                            failed = True
                            states.append([2])
                            init_frame = next_frame + self.skip_initialize
                    else:
                        states.append([0])
                    times.append(time.time() - start_time)
                    
                    if visualize:
                        if f == init_frame:
                            show_frame(image, anno[f],
                                       color=(0, 0, 255), pause=0.5)
                        else:
                            show_frame(image, state,
                                       color=(255, 0, 0), pause=0.001)
            
                # record tracking results
                self.record(r, seq_name, states, times)
