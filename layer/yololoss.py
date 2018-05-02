from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class YoloLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

   