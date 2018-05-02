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

if use_cuda:
    net = net.cuda()

criterion = YoloLoss()
optimizer = optim.SGD(net.parameters(),lr=arg.lr,mementum=0.9,weight_decay=1e-4)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx,(images,loc_targets,cls_targets,box_targets) in enumerate(train_Loader):
        images = Variable(images)
        loc_targets = Variable(loc_targets)
        cls_targets = Variable(cls_targets)
        box_targets = [Variable(x) for x in box_targets]
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda(0)
            box_targets = [x.cuda() for x in box_targets]
        
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output,loc_targets,cls_targets,box_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        print '%.3f %.3f' % (loss.data[0], train_loss/(batch_idx+1))


for epoch in range(start_epoch,start_epoch+200):
    train(epoch)
