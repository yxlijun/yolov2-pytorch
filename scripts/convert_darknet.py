# coding:utf-8 
import torch       
import torch.nn as nn        
import numpy as np        
from layer.model import Yolo
from data.config import cfg  

net = Yolo(cfg).darknet
darknet = np.load('./model/darknet19.weights.npz')
# layer1
conv_ids = [0,2,4,5,6,8,9,10,12,13,14,15,16]

for i,conv_id in enumerate(conv_ids):
    net.layer1[conv_id].conv.weight.data = torch.from_numpy(darknet['%d-convolutional/kernel:0' % i].transpose((3,2,0,1)))
    #net.layer1[conv_id].conv.bias.data = torch.from_numpy(darknet['%d-convolutional/biases:0' % i])
    net.layer1[conv_id].bn.weight.data = torch.from_numpy(darknet['%d-convolutional/gamma:0' % i])
    net.layer1[conv_id].bn.running_mean = torch.from_numpy(darknet['%d-convolutional/moving_mean:0' % i])
    net.layer1[conv_id].bn.running_var = torch.from_numpy(darknet['%d-convolutional/moving_variance:0' % i])

# layer2
conv_ids = [1,2,3,4,5]
for i,conv_id in enumerate(conv_ids):
    net.layer2[conv_id].conv.weight.data = torch.from_numpy(darknet['%d-convolutional/kernel:0' % (13+i)].transpose((3,2,0,1)))
    #net.layer2[conv_id].conv.bias.data = torch.from_numpy(darknet['%d-convolutional/biases:0' % (13+i)])
    net.layer2[conv_id].bn.weight.data = torch.from_numpy(darknet['%d-convolutional/gamma:0' % (13+i)])
    net.layer2[conv_id].bn.running_mean = torch.from_numpy(darknet['%d-convolutional/moving_mean:0' % (13+i)])
    net.layer2[conv_id].bn.running_var = torch.from_numpy(darknet['%d-convolutional/moving_variance:0' % (13+i)])
    
torch.save(net.state_dict(), './model/darknet.pth')