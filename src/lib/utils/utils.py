from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

class RunningAverageMeter(object):
    def __init__(self, alpha = 0.98):
        self.reset()
        self.alpha = alpha

    def reset(self):
        self.avg = 0.

    def update(self, val):
        if self.avg == 0. :
            self.avg = val
        else:
            self.avg = self.avg * self.alpha + (1 - self.alpha) * val


colors_table = [
    (255, 0, 0), #blue
    (0, 255, 0), # green
    (0,0,255), # red
    (255,255,255), # white
    (255,255,0), # yellow
    (160,32,240), # purple
    (255,165,0), # orange
    (139,126,102), # wheat
    (255,20,147), # deeppink
    (131,111,255) # Slateblue

]