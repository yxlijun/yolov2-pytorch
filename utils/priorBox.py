# -*- coding:utf-8 -*-
import torch  
from itertools import product
from data import *

class PriorBox(object):
    def __init__(self,cfg,fmsize=13):
        super(PriorBox, self).__init__()
        self.anchor = cfg.anchors
        self.fmsize = fmsize
    
    def forward(self):
        anchors = []
        anchor_num = len(self.anchor)
        for i,j in product(range(self.fmsize),repeat=2):
            cx = j
            cy = i
            for k in range(anchor_num):
                cw = self.anchor[k][0]
                ch = self.anchor[k][1]
                anchors+=[cx,cy,cw,ch]
        anchors = torch.Tensor(anchors).view(-1,4)
        return anchors


if __name__=='__main__':
    priorbox = PriorBox(cfg)
    print priorbox.forward()
        
                
        