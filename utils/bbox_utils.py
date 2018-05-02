# -*- coding:utf-8 -*-
from __future__ import division
import torch      


def prior_anchor_box(priorBox):
    boxes = priorBox.clone()
    xy = boxes[:,0:2]+0.5
    wh = boxes[:,2:]
    anchor_boxes = torch.cat((xy-wh/2,xy+wh/2),1)
    return anchor_boxes


def intersect(boxes1,boxes2):
    A = boxes1.size(0)
    B = boxes2.size(0)
    xymin = torch.max(boxes1[:,0:2].unsqueeze(1).expand(A,B,2),
                    boxes2[:,0:2].unsqueeze(0).expand(A,B,2))
    xymax = torch.min(boxes1[:,2:].unsqueeze(1).expand(A,B,2),
                    boxes2[:,2:].unsqueeze(0).expand(A,B,2))
    inter = torch.clamp((xymax-xymin),min=0)
    return inter[:,:,0]*inter[:,:,1]


def box_iou(boxes1,boxes2):
    '''
    boxes1: [A,4]
    boxes2: [B,4]
    return: [A,B]
    '''
    inter = intersect(boxes1,boxes2)
    area_a = ((boxes1[:,2]-boxes1[:,0])*(boxes1[:,3]-boxes1[:,1])).unsqueeze(1).expand_as(inter)
    area_b = ((boxes2[:,2]-boxes2[:,0])*(boxes2[:,3]-boxes2[:,1])).unsqueeze(0).expand_as(inter)
    union = area_a+area_b-inter
    return inter/union
    
    