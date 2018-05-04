# -*- coding:utf-8 -*- 
import torch  
import torch.nn as nn      
import torch.nn.functional as F       
from torch.autograd import Variable
from data import *
from utils.priorBox import PriorBox
from layer.detect import Detect
class ConvLayer(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size = 3):
        super(ConvLayer, self).__init__()
        padding = kernel_size//2 if kernel_size==3 else 0
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,padding=padding,bias=True)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self,x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class Darknet(nn.Module):
    def __init__(self,cfg):
        super(Darknet,self).__init__()
        self.layer1 = self._make_layer(cfg.cfg1,in_planes=3)
        self.layer2 = self._make_layer(cfg.cfg2,in_planes=512)

    def _make_layer(self,cfg,in_planes):
        layers = []
        for i in cfg:
            if i=='M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            elif isinstance(i,tuple):
                out_planes = i[0]
                layers += [ConvLayer(in_planes,out_planes,kernel_size=1)]
                in_planes = out_planes
            else:
                layers += [ConvLayer(in_planes,i)]
                in_planes = i
        return nn.Sequential(*layers)

    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x1,x2

class ReorgLayer(nn.Module):
    def __init__(self,stride=2):
        super(ReorgLayer,self).__init__()
        self.stride = stride
    
    def forward(self,x):
        B, C, H, W = x.size()
        s = self.stride
        x = x.view(B, C, H // s, s, W // s, s).transpose(3, 4).contiguous()
        x = x.view(B, C, H // s * W // s, s * s).transpose(2, 3).contiguous()
        x = x.view(B, C, s * s, H // s, W // s).transpose(1, 2).contiguous()
        return x.view(B, s * s * C, H // s, W // s)

class Yolo(nn.Module):
    def __init__(self,cfg):
        super(Yolo,self).__init__()
        self.cfg = cfg    
        self.darknet = Darknet(cfg)
        self.priorBox = Variable(PriorBox(cfg).forward(),volatile=True)
        self.conv1 = nn.Sequential(
            ConvLayer(1024,1024),
            ConvLayer(1024,1024)
        )
        self.conv2 = nn.Sequential(
            ConvLayer(512,64,kernel_size=1),
            ReorgLayer(2)
        )
        self.conv = nn.Sequential(
            ConvLayer(1280,1024),
            nn.Conv2d(1024, 5*(5+20), kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self,x,img_shape=None):
        x1,x2 = self.darknet(x)
        out1 = self.conv1(x2)
        out2 = self.conv2(x1)
        out = torch.cat((out1,out2),1)
        out = self.conv(out)
        b,c,h,w = out.size()        #[N,125,13,13]
        feat = out.permute(0, 2, 3, 1).contiguous().view(b, -1, self.cfg.anchor_num, self.cfg.class_num + 5)
        box_xy, box_wh = F.sigmoid(feat[..., 0:2]), feat[..., 2:4].exp()
        box_conf, score_pred = F.sigmoid(feat[..., 4:5]), feat[..., 5:].contiguous()
        box_prob = F.softmax(score_pred, dim=3)
        box_pred = torch.cat([box_xy, box_wh], 3)
        if self.training or img_shape is None:
            return out
        else:
            width,height = img_shape
            img_shape = Variable(torch.Tensor([[width, height, width, height]]))
            if cfg.use_cuda:
                self.priorBox, img_shape = self.priorBox.cuda(), img_shape.cuda()
            self.priorBox = self.priorBox.view(13*13,5,4).expand_as(box_pred)
            return Detect(self.cfg)(box_pred, box_conf, box_prob, self.priorBox, img_shape)

if __name__=='__main__':
    net = Yolo(cfg)
    print net.darknet.layer1[0].conv.weight.data
    #input = Variable(torch.randn(1,3,416,416))
    #out = net(input)
    #print out.size()
                

