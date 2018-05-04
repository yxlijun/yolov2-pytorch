# -*- coding:utf-8 -*-

import cv2
import os
import torch       
import torchvision
import  torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from data.config import cfg 
from layer.model import Yolo       

demo_path = './demo'
img_list = [os.path.join(demo_path,x) for x in os.listdir(demo_path) if x.endswith('.jpg')] 

result_path = os.path.join(demo_path,'result')
if not os.path.exists(result_path):
     os.mkdir(result_path)

net = Yolo(cfg)
net.load_state_dict(torch.load('./model/yolo-voc.pth'))
net.eval()
net.cuda()

transform = transforms.Compose([transforms.ToTensor()])
for img_path in img_list:
    image = cv2.imread(img_path)
    image_shape = (image.shape[1],image.shape[0])
    #print image_shape
    img= cv2.resize(image,(cfg.input_size,cfg.input_size))
    img = img[:,:,(2,1,0)]
    img = Variable(transform(img).unsqueeze(0)).cuda()
    boxes, scores, classes = net(img,image_shape)
    boxes, scores, classes = boxes.data.cpu(), scores.data.cpu(), classes.data.cpu()
    print('Find {} boxes for {}.'.format(len(boxes), img_path.split('/')[-1]))
    for i, c in list(enumerate(classes)):
        pred_class, box, score = cfg.VOC_CLASSES[c], boxes[i], scores[i]
        label = '{} {:.2f}'.format(pred_class, score)
        left,top,right,bottom = box
        textdata = str(pred_class)+":"+str(score)
        cv2.putText(image,textdata,(int(left),int(top-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        cv2.rectangle(image,(int(left),int(top)),(int(right),int(bottom)),(255,0,0),2)
    result_img = os.path.join(result_path,os.path.basename(img_path))
    cv2.imwrite(result_img,image)