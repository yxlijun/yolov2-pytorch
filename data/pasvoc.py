# -*- coding:utf-8 -*-
from __future__ import division
import os   
import cv2 
import torch.utils.data as data           
import torch       
import numpy as np            
from tools.encoder import DataEncoder
from utils.priorBox import PriorBox

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

    
class AnnotationTransform(object):
    def __init__(self, cfg):
        super(AnnotationTransform, self).__init__()
        self.class_to_id = dict(zip(cfg.VOC_CLASSES,range(len(cfg.VOC_CLASSES))))
    
    def __call__(self,target):
        result = []
        for obj in target.findall('object'):
            res = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin','ymin','xmax','ymax']
            for pt in pts:
                cur_pt = int(bbox.find(pt).text)
                res+=[cur_pt]
            res+=[self.class_to_id[name]]
            result+=[res]
        return result

class VOCDetection(data.Dataset):
    def __init__(self,root,image_set,cfg,transform=None,target_transform=None):
        super(VOCDetection,self).__init__()
        self.root = root
        self.image_set = image_set
        self.cfg = cfg
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = os.path.join('%s','Annotations','%s.xml')
        self._imgpath = os.path.join('%s','JPEGImages','%s.jpg')
        self.ids = list()
        self.data_encoder = DataEncoder(cfg)
        self.priorBox = PriorBox(cfg).forward()
        for (year,name) in self.image_set:
            rootpath = os.path.join(self.root,'VOC'+year)
            filepath = os.path.join(rootpath,'ImageSets','Main',name+'.txt')
            with open(filepath,'r') as f:
                for line in f.readlines():
                    self.ids.append((rootpath,line.strip()))

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.ids)

    def pull_item(self,index):
        img_id = self.ids[index]
        annopath = self._annopath%img_id
        imgpath = self._imgpath%img_id
        target = ET.parse(annopath).getroot()
        img = cv2.imread(imgpath)
        
        target = np.array(self.target_transform(target))

        boxes,labels = target[:,0:4],target[:,4]
        
        img,boxes,labels = self.transform(img,boxes,labels)
        loc_targets,cls_targets,box_targets = self.data_encoder.encode(boxes,labels,self.priorBox)
        return img,loc_targets,cls_targets,box_targets

def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), \
           torch.stack([x[1] for x in batch]), \
           torch.stack([x[2] for x in batch]), \
           [x[3] for x in batch]

if __name__=='__main__':
    from data.config import cfg
    from utils.YoloAugmentation import Yoloaugmentation
    image_set = [('2007','trainval')]
    dataset = VOCDetection(cfg.VOCroot,image_set,cfg,Yoloaugmentation(),AnnotationTransform(cfg))
    loc_targets,cls_targets,box_targets = dataset[1]
    print len(dataset),loc_targets.size(),cls_targets.size(),box_targets.size()
