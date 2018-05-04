# coding:utf-8

from __future__ import division
from torch.autograd import Function
from utils.bbox_utils import box_to_corners,filter_box,nms


class Detect(Function):
    def __init__(self, cfg):
        super(Detect, self).__init__()
        self.fm_size = cfg.fm_size
        self.nms_t, self.score_t = cfg.nms_threshold, cfg.score_threshold
    
    def forward(self,box_pred,box_conf,box_prob,priors,img_shape, max_boxes=10):
        box_pred[..., 0:2] += priors[..., 0:2]
        box_pred[..., 2:] *= priors[..., 2:]
        boxes = box_to_corners(box_pred) / self.fm_size
        boxes, scores, classes = filter_box(boxes, box_conf, box_prob, self.score_t)
        if boxes.numel()==0:
            return boxes,scores,classes
        boxes = boxes*img_shape.repeat(boxes.size(0),1)
        keep, count = nms(boxes, scores, self.nms_t,max_boxes)
        boxes = boxes[keep[:count]]
        scores = scores[keep[:count]]
        classes = classes[keep[:count]]
        return boxes, scores, classes
