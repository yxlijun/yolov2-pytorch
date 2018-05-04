# -*- coding:utf-8 -*- 

from easydict import EasyDict as edict
import os 

cfg = edict()

cfg.anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
cfg.anchor_num = 5

cfg.VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
cfg.class_num = len(cfg.VOC_CLASSES)

cfg.cfg1 = [32,'M',64,'M',128,(64,1),128,'M',256,(128,1),256,'M',512,(256,1),512,(256,1),512]
cfg.cfg2 = ['M',1024,(512,1),1024,(512,1),1024]

home = os.path.expanduser('~')
rdir = os.path.join(home,'data/VOCdevkit')

cfg.VOCroot = rdir

cfg.input_size = 416

cfg.batch_size = 16

cfg.fm_size = 13

cfg.nms_threshold = 0.4
cfg.score_threshold = 0.5
cfg.iou_threshold = 0.4

cfg.use_cuda=True   
