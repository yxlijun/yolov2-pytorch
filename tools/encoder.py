# -*- coding:utf-8 -*-
from __future__ import division
import torch
from utils.bbox_utils import prior_anchor_box,box_iou

class DataEncoder(object):
    def __init__(self,cfg):
        super(DataEncoder, self).__init__()
        self.anchors = cfg.anchors
        self.input_size = cfg.input_size

    def encode(self,boxes,labels,priorBox):
        num_boxes = len(boxes)
        fmsize = int(self.input_size / 32)   #13
        grid_size = int(self.input_size /fmsize)    #32
        boxes*=self.input_size
        bx = (boxes[:,0]+boxes[:,2])*0.5 / grid_size
        by = (boxes[:,1]+boxes[:,3])*0.5 / grid_size
        bw = (boxes[:,2]-boxes[:,0]) / grid_size
        bh = (boxes[:,3]-boxes[:,1]) / grid_size

        tx = bx - torch.floor(bx)
        ty = by - torch.floor(by)

        anchor_boxes = prior_anchor_box(priorBox)

        anchor_boxes = anchor_boxes.view(fmsize,fmsize,5,4)

        ious = box_iou(anchor_boxes.view(-1,4),boxes/grid_size)    #[13*13*5,num_boxes]
        ious = ious.view(fmsize,fmsize,5,num_boxes)
        loc_targets = torch.zeros(5,4,fmsize,fmsize)
        cls_targets = torch.zeros(5,20,fmsize,fmsize)

        for i in range(num_boxes):
            cx = int(bx[i])
            cy = int(by[i])
            _,max_idx = ious[cx,cy,:,i].max(0)
            j = max_idx[0]
            cls_targets[j,labels[i],cy,cx] = 1

            tw = bw[i] / self.anchors[j][0]
            th = bh[i] / self.anchors[j][1]
            loc_targets[j,:,cy,cx] = torch.Tensor([tx[i],ty[i],tw,th])
        return loc_targets,cls_targets,boxes/grid_size
        
        
