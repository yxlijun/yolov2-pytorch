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
from data.pasvoc import VOCDetection,AnnotationTransform,collate_fn
from utils.YoloAugmentation import Yoloaugmentation_test
from  torch.utils.data import DataLoader

image_set = [('2007','test')]
test_dataset = VOCDetection(cfg.VOCroot,image_set,cfg,Yoloaugmentation_test(),AnnotationTransform(cfg))
test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size,
                        shuffle=False,num_workers =4,collate_fn =collate_fn)

cachedir = './cachedir'
if not os.path.exists(cachedir):
     os.mkdir(cachedir)

cachedir_predict = os.path.join(cachedir,'predict_dir')
if not os.path.exists(cachedir_predict):
    os.mkdir(cachedir_predict)

det_list = [os.path.join(cachedir_predict,file) for file in os.listdir(cachedir_predict)]
for det_class_file in det_list:
    with open(det_class_file,mode='w') as f:
        pass

def test():
    net = Yolo(cfg)
    net.load_state_dict(torch.load('./model/yolo-voc.pth'))
    net.eval()
    net.cuda()
    print 'load model finish'
    num_images = len(test_dataset)
    transform = transforms.Compose([transforms.ToTensor()])
    for i in range(num_images):
        image,imagename = test_dataset.pull_image(i)
        image_shape = (image.shape[1],image.shape[0])
        img= cv2.resize(image,(cfg.input_size,cfg.input_size))
        img = img[:,:,(2,1,0)]
        img = Variable(transform(img).unsqueeze(0)).cuda()
        boxes, scores, classes = net(img,image_shape)
        boxes, scores, classes = boxes.data.cpu(), scores.data.cpu(), classes.data.cpu()
        print('Find {} boxes for {}.'.format(len(boxes), imagename))
        for i,c in list(enumerate(classes)):
            pred_class, box, score = cfg.VOC_CLASSES[c], boxes[i], scores[i]
            filename = os.path.join(cachedir_predict,'det_test_'+pred_class+'.txt')
            with open(filename,mode='a') as f:
                left,top,right,bottom = box
                content = imagename+' '+str(score)+' '+str(int(left))+' '+str(int(top))+' '+str(int(right))+' '+str(int(bottom))+'\n'
                f.write(content)

if __name__=='__main__':
    test()