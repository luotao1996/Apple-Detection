
# Apple Detection and Estimated the distance by one Camera
This is a project that can detect apples and estimate the distance from object to camera
The master branch works with tensorpack 0.9.0.1 <br/>

## Main References
+ [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
+ [单目视觉定位测距的两种方式](https://blog.csdn.net/sillykog/article/details/71214107)
+ [基于单目视觉的实时测距方法研究](https://wenku.baidu.com/view/afe91df8941ea76e58fa0415.html)
+ [线性回归分析法](https://wenku.baidu.com/view/88ddce59770bf78a6429549e.html)

## Dependencies
+ Python 3.3+; OpenCV
+ TensorFlow ≥ 1.6
+ Tensorpack = 0.9.0.1
+ pycocotools: `pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
+ Pre-trained [ResNet model](http://models.tensorpack.com/FasterRCNN/)
  from tensorpack model zoo
+ [COCO data](http://cocodataset.org/#download). It needs to have the following directory structure:
```
COCO/DIR/
  annotations/
    instances_train201?.json
    instances_val201?.json
  train201?/
    COCO_train201?_*.jpg
  val201?/
    COCO_val201?_*.jpg
```
+ [Apple-100 data](https://pan.baidu.com/). :
```
Apple images/
  1-30-30cm.jpg
  2-30-30cm.jpg
  ......
  6-50-30cm.jpg
  7-50-30cm.jpg
  ......
  11-70-30cm.jpg
  12-70-30cm.jpg
  ......
```
## Usage
### Train:

To train on a single machine:
```
./train.py --config \
    DATA.BASEDIR=/root/datasets/COCO/DIR
    BACKBONE.WEIGHTS=/root/datasets/COCO-R50C4-MaskRCNN-Standard.npz
    TRAIN.LR_SCHEDULE=[150000,230000,280000]
```
### Inference:
To predict on an image (needs DISPLAY to show the outputs):
```
./start.py 
--image
/path/to/input.jpg
--load
/path/to/COCO-R50C4-MaskRCNN-Standard.npz

```
## Results
The models are fine-tuned from ResNet pre-trained R50C4 models in
[tensorpack model zoo](http://models.tensorpack.com/FasterRCNN/)

Performance in [Apple-100 datasets](http://https://pan.baidu.com/) can
be approximately reproduced.
   
    |      Distance     |     Distance error(≈5%)  |  Mean square error(<1) | 
    |        30         |        29.9              |     0.42146            |    
    |        50         |        49.2              |     0.5892             |
    |        70         |        71.1              |     0.532              |
    |        90         |        94.3              |     0.73477            |
    |        120        |        115.8             |     0.99904            |
   
 
## Some examples

Here are some visualization results of the apple detection model.<br/>
![Image text](https://github.com/luotao1996/Apple_Detection/blob/master/examples/apple1.png)<br/>
![Image text](https://github.com/luotao1996/Apple_Detection/blob/master/examples/apple2.png)<br/>
![Image text](https://github.com/luotao1996/Apple_Detection/blob/master/examples/apple3.png)

