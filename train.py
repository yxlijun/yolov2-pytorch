# -*- coding:utf-8 -*-
import torch
import torch.optim as optim
from torch.autograd import Variable

import os  
import argparse   

from layer.model import Yolo
from layer.yololoss import YoloLoss
from data.config import cfg 
from data.pasvoc import AnnotationTransform,VOCDetection,collate_fn
from utils.YoloAugmentation import Yoloaugmentation




use_cuda = torch.cuda.is_available()
train_set = [('2007','trainval')]
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

print '==> Preparing data..'
train_dataset = VOCDetection(cfg.VOCroot,train_set,cfg,Yoloaugmentation(),AnnotationTransform(cfg))
train_Loader = torch.utils.data.DataLoader(train_dataset,batch_size=cfg.batch_size,
                                            shuffle=True,num_workers =4,collate_fn =collate_fn)

    
# Model
net = Yolo(cfg)


criterion = YoloLoss()
optimizer = optim.SGD(net.parameters(),lr=arg.lr,mementum=0.9,weight_decay=1e-4)

def train():
    


