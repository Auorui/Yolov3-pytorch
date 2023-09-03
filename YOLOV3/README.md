## YOLOV3：You Only Look Once目标检测模型在Pytorch当中的实现
---

## 环境搭建
torch2.0以上会遇到:
* NotImplementedError: Could not run 'torchvision::nms' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'torchvision::nms' is only available for these backends: [CPU, QuantizedCPU, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMeta, Tracer, AutocastCPU, AutocastCUDA, FuncTorchBatched, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PythonDispatcher].

大致是torch里面的nms在2.0以上无发找到，估计是个bug。

目前在torch1.9中成功运行:

### 从头开始搭建环境

**1. 虚拟环境创建：conda create -n torch1.9 python=3.9**

torch1.9安装指令：pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html  -i https://pypi.tuna.tsinghua.edu.cn/simple

最好使用pip安装，conda指令有些问题。

**2.【如果遇到了AttributeError: 'ImageDraw' object has no attribute 'textsize'】**

可以降低pillow版本  pip install Pillow==9.5.0

**3. 这里用到了可视化，所以要下载tensorboard**

pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

**4. mediapipe是pyzjr的一个依赖库，所以还是先装上它**

pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple

**5、安装pyzjr**

pip install pyzjr -i https://pypi.tuna.tsinghua.edu.cn/simple

1.1.6版本可用。

## 文件下载
训练所需的yolo_weights.pth可以在百度网盘下载。  
链接:https://pan.baidu.com/s/12ZfwTF3lEGq2Iqyscv5kIg?pwd=4258 
提取码:4258


## 训练步骤
### a、训练自己的数据集
#### 1. 数据集的准备
   
这里自行准备，目录结构可以调用pyzjr查看

````python
import pyzjr.dlearn.voc as voc

voc.voc_catalog()
````

控制台输出：
````
VOC Catalog:
----------------------------------------------------------------------
VOCdevkit
    VOC2007
        -ImageSets/Segmentation    Store training index files
        -JPEGImages                Store image files
        -SegmentationClass         Store label files
----------------------------------------------------------------------
````

可以看到这里，训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

model_data下的classes.txt文件一定要修改，要与你自己检测的目标对应才行，比如我这里有三类：
````
spalling
break
rebar
````

#### 3. 数据集的处理   
修改voc_yolo_txt.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

#### 4. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。如果需要修改的地方，可以查看trainconfig.py文件，里面就是一些训练参数的修改，
部分是不需要修改的。不明白的地方可以查看下面的help。

![image](https://github.com/Auorui/Yolov3-pytorch/assets/100789256/47ca5176-a526-4837-a979-07ed53fb3b24)


#### 5. 训练结果预测   
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。

需要去yolo.py里面修改model_path以及classes_path，这两个参数必须要修改。   

**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**   

完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

#### 6. 注意，文件夹存放的位置可能不同，相对应导入的包会报错，了解一下相对路径和绝对路径就知道了

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载yolo_weights.pth，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolo_weights.pth',
    "classes_path"      : 'model_data/classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [416, 416],
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py，输入文件图片的路径进行预测  
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  


