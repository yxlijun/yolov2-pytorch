# -*- coding:utf-8 -*- 
from __future__ import division
import torchvision.transforms as transforms
import torch         
import numpy as np        
from data import *
import cv2
from numpy import random

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms
    
    def __call__(self,img,boxes=None,lables=None):
        for t in self.transforms:
            img,boxes,lables = t(img,boxes,lables)
        return img,boxes,lables


class ConvertFromInts(object):
    def __call__(self,img,boxes=None,lables=None):
        return np.array(img,dtype=np.float32),boxes.astype(np.float32),lables


class SubtractMeans(object):
    def __init__(self,mean):
        self.mean = np.array(mean,dtype=np.float32)
    
    def __call__(self,img,boxes=None,lables=None):
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32),boxes,lables


class Resize(object):
    def __init__(self,cfg):
        self.input_size = cfg.input_size

    def __call__(self,img,boxes=None,lables=None):
        img = cv2.resize(img, (self.input_size,self.input_size))
        return img.astype(np.float32),boxes,lables


class ToPercentCoords(object):
    def __call__(self,img,boxes=None,lables=None):
        height,width,channels = img.shape
        boxes[:,0] /= width
        boxes[:,1] /= height
        boxes[:,2] /= width
        boxes[:,3] /= height
        return img,boxes,lables

class ToTensor(object):
    def __call__(self,img,boxes=None,lables=None):
        return torch.from_numpy(img),torch.from_numpy(boxes),torch.from_numpy(lables)



class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels





class Yoloaugmentation(object):
    def __init__(self,mean=(104,117,123)):
        super(Yoloaugmentation, self).__init__()
        self.mean = mean
        self.augment = Compose([
            ConvertFromInts(),
            RandomSampleCrop(),
            ToPercentCoords(),
            Resize(cfg),
            SubtractMeans(self.mean),
            ToTensor()
        ])

    def __call__(self,img,boxes,labels):
        return self.augment(img,boxes,labels)
