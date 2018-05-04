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
from utils.YoloAugmentation import Yoloaugmentation_train,Yoloaugmentation_test


parser = argparse.ArgumentParser(description='train yolov2-pytorch')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--resume','-r',action="store_true",help='resume from checkoutput')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
train_set = [('2007','trainval')]
test_set = [('2007','test')]
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
step_epoch = [100,150,180]

print '==> Preparing data..'
train_dataset = VOCDetection(cfg.VOCroot,train_set,cfg,Yoloaugmentation_train(),AnnotationTransform(cfg))

test_dataset = VOCDetection(cfg.VOCroot,test_set,cfg,Yoloaugmentation_test(),AnnotationTransform(cfg))

train_Loader = torch.utils.data.DataLoader(train_dataset,batch_size=cfg.batch_size,
                                            shuffle=True,num_workers =4,collate_fn =collate_fn)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=cfg.batch_size,
                                            shuffle=False,num_workers=4,collate_fn=collate_fn)


# Model
net = Yolo(cfg)
if args.resume:
    print('from checkpoint resume')
    checkpoint = torch.load('./checkpoint/yolov2_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    net.darknet.load_state_dict(torch.load('./model/darknet.pth'))

if use_cuda:
    net = net.cuda()

criterion = YoloLoss()
optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)

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

def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx,(images,loc_targets,cls_targets,box_targets) in enumerate(test_loader):
        images = Variable(images)
        loc_targets = Variable(loc_targets)
        cls_targets = Variable(cls_targets)
        box_targets = [Variable(x) for x in box_targets]
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda(0)
            box_targets = [x.cuda() for x in box_targets]

        output = net(images)
        loss = criterion(output,loc_targets,cls_targets,box_targets)
        test_loss += loss.data[0]
        print '%.3f %.3f' % (loss.data[0], test_loss/(batch_idx+1))

    global best_loss
    test_loss /= len(test_loader)
    if test_loss<best_loss:
        print('\nSaving model')
        state = {
            'net':net.state_dict(),
            'loss':test_loss,
            'epoch':epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,'./checkpoint/yolov2_ckpt.pth')
        torch.save(net.state_dict(),'./model/yolov2.pth')
        best_loss = test_loss


def adjust_learning_rate(epoch):
    if epoch in step_epoch:
        args.lr = 0.1*args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    


if __name__=='__main__':
    for epoch in range(start_epoch,start_epoch+200):
        adjust_learning_rate(epoch)
        train(epoch)
        test(epoch)
