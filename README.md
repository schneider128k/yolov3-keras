# YOLOv3 in Keras

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

This is my Keras implementation of YOLOv3 (Tensorflow backend) that is based on [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).

I have written a Lambda layer for decoding the output produced by the YOLOv3 model (it is no longer necessary to run a 
tensorflow session as in qqwweee's implementation). 

I have also implemented a (fast) non-maximum-suppression based on 
[jrosebr1/imutils](https://github.com/jrosebr1/imutils/blob/master/imutils/object_detection.py).

So far I have only implemented detection. I am working on the code for training. 
My goal is to have an elegant implementation of YOLOv3 in Keras for my upcoming 
undergraduate course on Artificial Intelligence in Fall 2019. I have attempted to use tensor names 
that are as close as possible to the names used in the paper 
[YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf). Compare section 2.1 
Bounding Box Prediction/Figure 2 and the implementation of the function ```make_decoder_layer``` in 
```yolov3\model.py```. 

## Quick start

* Download YOLOv3 weights ```yolov3.weights``` from [YOLO website](https://pjreddie.com/media/files/yolov3.weights).
* Convert the Darknet YOLOv3 model to a Keras model:
```python convert.py yolov3.cfg yolov3.weights model_data/yolov3.h5```
* Run YOLOv3 detection: ```python detection.py --image <path to image>``` or 
```python detection.py --video <path to video>``` 

The video [YOLOv3 object detection applied to ArmA3](https://www.youtube.com/watch?v=Nrg5WcMN9lU) shows this implementation in action.

Note that it takes a while to load the Keras model ```model_data\yolov3.h5``` before the detection 
starts.  