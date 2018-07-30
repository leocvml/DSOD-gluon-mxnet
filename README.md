# DSOD-gluon-mxnet



this repo attemps to reproduce [DSOD: Learning Deeply Supervised Object Detectors from Scratch](https://arxiv.org/abs/1708.01241) use gluon reimplementation 

## Abstract ##

**The  DSOD method is a multi-scale proposal-free detection framework similar to SSD** 


Train detection model **from scratch**.

State-of-the-art object objectors rely heavily on the off-the-shelf networks pre-trained on large-scale classification datasets like ImageNet, which incurs learning bias due to the difference on both the loss functions and the category distributions between classification and detection tasks.

## quick start  (very easy) ##
```
1. clone (download)
2. execution 'getpikachu.py' get dataset
3. run 'DSOD.py' your will see result( no optimization)
```
## Requirements ##
```
mxnet 0.11.0
```
## Network Arch ##
![](https://i.imgur.com/BW2ze1B.png)![](https://github.com/leocvml/DSOD-gluon-mxnet/blob/master/backbone_fisrthalf.PNG)
```python
####################################################
###
###  num of  channels in the 1st conv,
###  num of layer in 1st conv
###  growth rate,
###  factor in transition layer)
###  num_class  ( class + 1)
###################################################
class DSOD(nn.HybridBlock):
    def __init__(self,stem_filter, num_init_layer, growth_rate, factor,num_class):
        if factor == 0.5:
            self.factor = 2
        else:
            self.factor = 1
        self.num_cls = num_class
        self.sizes = [[.2, .2], [.37,.37],[.45,.45], [.54,.54], [.71,.71], [.88,.88]]  #
        self.ratios = [[1,2,0.5]]*6
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        trans1_filter = ((stem_filter * 2) + (num_init_layer * growth_rate) //self.factor )
        super(DSOD, self).__init__()
        self.backbone_fisrthalf = nn.HybridSequential()
        with self.backbone_fisrthalf.name_scope():
            self.backbone_fisrthalf.add(
                stemblock(stem_filter),
                DenseBlcok(6, growth_rate),
                transitionLayer(trans1_filter),
                DenseBlcok(8, growth_rate)

            )
        trans2_filter = ((trans1_filter) + (8 * growth_rate) //self.factor )
        trans3_filter = ((trans2_filter) + (8 * growth_rate) //self.factor )


        self.backbone_secondehalf = nn.HybridSequential()
        with self.backbone_secondehalf.name_scope():
            self.backbone_secondehalf.add(
                transitionLayer(trans2_filter),
                DenseBlcok(8, growth_rate),
                transitionLayer(trans3_filter,with_pool=False),
                DenseBlcok(8, growth_rate),
                transitionLayer(256, with_pool=False)
            )
        self.PC_layer = nn.HybridSequential()   # pool -> conv
        numPC_layer =[256,256,128,128,128]
        with self.PC_layer.name_scope():
            for i in range(5):
                self.PC_layer.add(
                    pool_conv(numPC_layer[i]),
                )
        self.CC_layer = nn.HybridSequential() # conv1 -> conv3
        numCC_layer = [256,128,128,128]
        with self.CC_layer.name_scope():
            for i in range(4):
                self.CC_layer.add(
                    conv_conv(numCC_layer[i])
                )

        self.class_predictors = nn.HybridSequential()
        with self.class_predictors.name_scope():
            for _ in range(6):
                self.class_predictors.add(
                        cls_predictor(self.num_anchors,self.num_cls)
                )

        self.box_predictors = nn.HybridSequential()
        with self.box_predictors.name_scope():
            for _ in range(6):
                self.box_predictors.add(
                    bbox_predictor(self.num_anchors)
                )

    def flatten_prediction(self,pred):
        return pred.transpose(axes=(0, 2, 3, 1)).flatten()

    def concat_predictions(self,preds):
        return nd.concat(*preds, dim=1)

    def hybrid_forward(self, F, x):

        anchors, class_preds, box_preds = [], [], []

        scale_1 = self.backbone_fisrthalf(x)

        anchors.append(MultiBoxPrior(
            scale_1, sizes=self.sizes[0], ratios=self.ratios[0]))
        class_preds.append(
            self.flatten_prediction(self.class_predictors[0](scale_1)))
        box_preds.append(
            self.flatten_prediction(self.box_predictors[0](scale_1)))


        out = self.backbone_secondehalf(scale_1)
        PC_1 = self.PC_layer[0](scale_1)
        scale_2 = F.concat(out,PC_1,dim=1)

        anchors.append(MultiBoxPrior(
            scale_2, sizes=self.sizes[1], ratios=self.ratios[1]))
        class_preds.append(
            self.flatten_prediction(self.class_predictors[1](scale_2)))
        box_preds.append(
            self.flatten_prediction(self.box_predictors[1](scale_2)))

        scale_predict = scale_2
        for i in range(1,5):

            PC_Predict = self.PC_layer[i](scale_predict)
            CC_Predict = self.CC_layer[i-1](scale_predict)
            scale_predict = F.concat(PC_Predict, CC_Predict, dim=1)

            anchors.append(MultiBoxPrior(
                scale_predict, sizes=self.sizes[i+1], ratios=self.ratios[i+1]))
            class_preds.append(
                self.flatten_prediction(self.class_predictors[i+1](scale_predict)))
            box_preds.append(
                self.flatten_prediction(self.box_predictors[i+1](scale_predict)))

           # print(scale_predict.shape)

        anchors = self.concat_predictions(anchors)
        class_preds = self.concat_predictions(class_preds)
        box_preds = self.concat_predictions(box_preds)

        class_preds = class_preds.reshape(shape=(0, -1, self.num_cls+1))

        return anchors, class_preds, box_preds

net = nn.HybridSequential()
####################################################
###
###  num of  channels in the 1st conv,
###  num of layer in 1st conv
###  growth rate,
###  factor in transition layer)
###  num_class  ( class + 1)
###################################################
with net.name_scope():
    net.add(
        DSOD(32, 6, 48, 1, 1)  # 64 6 48 1 1 in paper
    )

```





## training dataset ##
this repo is training on pikachu dataset
**get pikachu dataset**
```python
from mxnet.test_utils import download
import os.path as osp
def verified(file_path, sha1hash):
    import hashlib
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    matched = sha1.hexdigest() == sha1hash
    if not matched:
        print('Found hash mismatch in file {}, possibly due to incomplete download.'.format(file_path))
    return matched

url_format = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/{}'
hashes = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
          'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
          'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
for k, v in hashes.items():
    fname = k
    target = osp.join('data', fname)
    url = url_format.format(k)
    if not osp.exists(target) or not verified(target, v):
        print('Downloading', target, url)
        download(url, fname=fname, dirname='data', overwrite=True)
```

## how to train your own dataset  ##
**first make your dataset to .rec 
you can check my another repo 
https://github.com/leocvml/mxnet-im2rec_tutorial**
## parameter setting  ##
```python
######################################################
##
##
## parameter setting 
## set image size, batchsize
## training( without retrain no inference ) : retrain =False, inference =False, epoch = number of epoch
## training( with retrain and inference after training) : retrain =True, inference =True, inference_Data = name of image, epoch= number
## only inference ( load weighting and inference) : retrain =True, inference =True, inference_data = name of image, epoch = 0
#####################################################
data_shape = 512
batch_size = 4
rgb_mean = nd.array([123, 117, 104])
retrain = True
inference = True
inference_data = 'pikachu.jpg'
epoch = 0
```
## result ##
**i use pikachu dataset(from gluon tutorial) this result didn't optimization**
**you can change anchor size ,Bigger network ,Add hidden layer,Long training time,NMS thresholding , hard negative mining etc**
![](https://github.com/leocvml/DSOD-gluon-mxnet/blob/master/detection2.PNG)

## learn more .. ##
you can also see these  tutorial by gluon team,
Learn more about SSD and other detection model

chinese:
https://zh.gluon.ai/chapter_computer-vision/ssd.html

english:
https://gluon.mxnet.io/chapter08_computer-vision/object-detection.html


## Note  ##

this result didn't optimization
**fix bug on 2018/07/30**

**I appreciate the author's effort in providing a nice experiment in this paper**

**very thanks mxnet gluon team, they build the very nice tutorial for everyone**


