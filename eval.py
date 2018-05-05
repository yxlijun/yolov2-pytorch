#coding:utf-8 
import os           
from eval.voc_eval import voc_eval
from data.config import cfg
import numpy as np          

cachedir_predict = './cachedir/predict_dir'
cachedir = './cachedir'
imageset = ('2007','test')

if not os.path.exists(cachedir):
    os.mkdir(cachedir)

det_list = [os.path.join(cachedir_predict,file) for file in os.listdir(cachedir_predict)]
det_classes = list()
for file in det_list:
    classes = os.path.splitext(os.path.basename(file))[0].split('_')[-1]
    det_classes.append(classes)
    detpath = file.replace(classes,'%s')


annopath = os.path.join(cfg.VOCroot,'VOC'+imageset[0],'Annotations','%s.xml')
imagesetfile = os.path.join(cfg.VOCroot,'VOC'+imageset[0],'ImageSets','Main',imageset[1]+'.txt')

MAPList = list()
for classname in det_classes:
    rec,prec,ap = voc_eval(detpath,annopath,imagesetfile,classname,cachedir)
    print '%s\t AP:%.4f' %(classname,ap)
    MAPList.append(ap)

Map = np.array(MAPList)
mean_Map = np.mean(Map)
print '------ Map: %.4f' %(mean_Map)