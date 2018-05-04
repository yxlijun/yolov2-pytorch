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
    

def box_to_corners(boxes):
    box_mins = boxes[...,0:2]-(boxes[...,2:4]*0.5)
    box_maxs = boxes[...,0:2]+(boxes[...,2:4]*0.5)
    return torch.cat([box_mins,box_maxs],3)

def filter_box(boxes, box_conf, box_prob, threshold=.5):
    box_scores = box_conf.repeat(1, 1, 1, box_prob.size(3)) * box_prob
    box_class_scores, box_classes = torch.max(box_scores, dim=3)
    prediction_mask = box_class_scores > threshold
    prediction_mask4 = prediction_mask.unsqueeze(3).expand(boxes.size())

    boxes = torch.masked_select(boxes, prediction_mask4).contiguous().view(-1, 4)
    scores = torch.masked_select(box_class_scores, prediction_mask)
    classes = torch.masked_select(box_classes, prediction_mask)
    return boxes, scores, classes

def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel()== 0:
        return keep
    x1 = boxes[:,0]
    x2 = boxes[:,2]
    y1 = boxes[:,1]
    y2 = boxes[:,3]
    area = torch.mul(x2-x1,y2-y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel()>0:
        i = idx[-1]
        keep[count] = i
        count+=1
        if idx.size(0)==1:
            break 
        idx = idx[:-1]
        torch.index_select(x1,0,idx,out=xx1)
        torch.index_select(y1,0,idx,out=yy1)
        torch.index_select(x2,0,idx,out=xx2)
        torch.index_select(y2,0,idx,out=yy2)

        xx1 = torch.clamp(xx1,min=x1[i])
        yy1 = torch.clamp(yy1,min=y1[i])
        xx2 = torch.clamp(xx2,max=x2[i])
        yy2 = torch.clamp(yy2,max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = torch.clamp(xx2-xx1,min=0.0)
        h = torch.clamp(yy2-yy1,min=0.0)
        inter = w*h
        rem_areas = torch.index_select(area,0,idx)
        iou = inter/(area[i]+rem_areas-inter)
        idx = idx[iou.le(overlap)]

    return keep,count
