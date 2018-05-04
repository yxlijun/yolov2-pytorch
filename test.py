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

net = Yolo(cfg)
net.load_state_dict(torch.load('./model/yolov2.pth'))
net.eval()
net.cuda()

transform = transforms.Compose([transforms.ToTensor()])
for img_path in img_list:
    image = cv2.imread(img_path)
    img= cv2.resize(image,(cfg.input_size,cfg.input_size))
    img = Variable(transform(img).unsqueeze(0)).cuda()
    boxes, scores, classes = net(img,image.size)
    boxes, scores, classes = boxes.data.cpu(), scores.data.cpu(), classes.data.cpu()
    print('Find {} boxes for {}.'.format(len(boxes), img_path.split('/')[-1]))
    # for i, c in list(enumerate(classes)):
    #     pred_class, box, score = cfg.VOC_CLASSES[c], boxes[i], scores[i]
    #     label = '{} {:.2f}'.format(pred_class, score)
    #     draw_box(cfg, img, label, box, c)
    # image.save(os.path.join(save_path, img.split('/')[-1]), quality=90)


def draw_box(cfg, image, label, box, c):
    w, h = image.size
    font = ImageFont.truetype(font='./config/FiraMono-Medium.otf', size=np.round(3e-2 * h).astype('int32'))
    thickness = (w + h) // 300
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)
    left, top, right, bottom = box
    top, left = max(0, np.round(top).astype('int32')), max(0, np.round(left).astype('int32'))
    right, bottom = min(w, np.round(right).astype('int32')), min(h, np.round(bottom).astype('int32'))
    print(label, (left, top), (right, bottom))
    text_orign = np.array([left, top - label_size[1]]) if top - label_size[1] >= 0 else np.array([left, top + 1])
    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=cfg.colors[c])
    draw.rectangle([tuple(text_orign), tuple(text_orign + label_size)], fill=cfg.colors[c])
    draw.text(text_orign, label, fill=(0, 0, 0), font=font)
    del draw

def draw_box(cfg,img,label,box,c):
