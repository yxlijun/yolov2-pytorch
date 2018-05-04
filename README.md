##yolov2 implement pytorch ##

this is a pytorch implementation of yolov2,including train and test phase.

## Installation and demo ##
1.Clone this repository

    git clone https://github.com/yxlijun/yolov2-pytorch
2.Download Darknet model cfg and weights from the official YOLO website.

    wget http://pjreddie.com/media/files/yolo-voc.weights
	wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo-voc.cfg

3.Convert the weights to .pth
    
	python tools/darknet_to_pytorch.py

4.demo and test

    python test.py

## train 
1.Download the training, validation, test data and VOCdevkit
    
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

2.Extract all of these tars into one directory named VOCdevkit
    
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
3.put dataset into /home/xxx/data/    directory

4.Run the training program: 
    
	python train.py.

## result show ##
 ![result show](https://github.com/yxlijun/yolov2-pytorch/blob/master/demo/result/dog.jpg)