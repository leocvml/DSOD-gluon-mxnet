
import mxnet as mx
from matplotlib import pyplot as plt
from mxnet.gluon import nn
from mxnet.contrib.ndarray import MultiBoxPrior
from mxnet.contrib.ndarray import MultiBoxDetection
from mxnet.contrib.ndarray import MultiBoxTarget
import numpy as np
from mxnet import gluon
from mxnet import metric
import time
from mxnet import nd
from mxnet import image


# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--epoch", help="number of epochs", type=int)
# parser.add_argument("--retrain", help="load weighting and continue training", type=int)
# parser.add_argument("--test", help="inference", type=int)
# parser.add_argument("--testimg", help="test image", type=str)
#
# args = parser.parse_args()

def training_targets(anchors, class_preds, labels):
    class_preds = class_preds.transpose(axes=(0, 2, 1))

    return MultiBoxTarget(anchors, labels, class_preds,
                          overlap_threshold=0.3)  # ,overlap_threshold=0.3,negative_mining_ratio=0.3


class stemblock(nn.HybridBlock):
    def __init__(self, filters):
        super(stemblock, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2D(self.filters, kernel_size=3, padding=1, strides=2)
        self.bn1 = nn.BatchNorm()
        self.act1 = nn.Activation('relu')

        self.conv2 = nn.Conv2D(self.filters, kernel_size=1, strides=1)
        self.bn2 = nn.BatchNorm()
        self.act2 = nn.Activation('relu')

        self.conv3 = nn.Conv2D(self.filters * 2, kernel_size=1, strides=1)

        self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=2)

    def hybrid_forward(self, F, x):
        stem1 = self.act1(self.bn1(self.conv1(x)))
        stem2 = self.act2(self.bn2(self.conv2(stem1)))
        stem3 = self.conv3(stem2)
        out = self.pool(stem3)
        return out


class conv_block(nn.HybridBlock):
    def __init__(self, filters):
        super(conv_block, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=1),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=3, padding=1)
            )

    def hybrid_forward(self, F, x):
        return self.net(x)


class DenseBlcok(nn.HybridBlock):
    def __init__(self, num_convs, num_channels):  # layers, growth rate
        super(DenseBlcok, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            for _ in range(num_convs):
                self.net.add(
                    conv_block(num_channels)
                )

    def hybrid_forward(self, F, x):
        for blk in self.net:
            Y = blk(x)
            x = F.concat(x, Y, dim=1)

        return x


class transitionLayer(nn.HybridBlock):
    def __init__(self, filters, with_pool=True):
        super(transitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm()
        self.act1 = nn.Activation('relu')
        self.conv1 = nn.Conv2D(filters, kernel_size=1)
        self.with_pool = with_pool
        if self.with_pool:
            self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=2)

    def hybrid_forward(self, F, x):
        out = self.conv1(self.act1(self.bn1(x)))
        if self.with_pool:
            out = self.pool(out)
        return out


class conv_conv(nn.HybridBlock):
    def __init__(self, filters):
        super(conv_conv, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=1, strides=1),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=3, strides=2, padding=1)
            )

    def hybrid_forward(self, F, x):
        return self.net(x)


class pool_conv(nn.HybridBlock):
    def __init__(self, filters):
        super(pool_conv, self).__init__()
        self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=2)
        self.bn = nn.BatchNorm()
        self.act = nn.Activation('relu')
        self.conv = nn.Conv2D(filters, kernel_size=1)

    def hybrid_forward(self, F, x):
        out = self.conv(self.act(self.bn(self.pool(x))))
        return out


class cls_predictor(nn.HybridBlock):  # (num_anchors * (num_classes + 1), 3, padding=1)
    def __init__(self, num_anchors, num_classes):
        super(cls_predictor, self).__init__()
        self.class_predcitor = nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

    def hybrid_forward(self, F, x):
        return self.class_predcitor(x)


class bbox_predictor(nn.HybridBlock):  # (num_anchors * 4, 3, padding=1)
    def __init__(self, num_anchors):
        super(bbox_predictor, self).__init__()
        self.bbox_predictor = nn.Conv2D(num_anchors * 4, 3, padding=1)

    def hybrid_forward(self, F, x):
        return self.bbox_predictor(x)


# stemfilter = 64
# num_init_layer = 6
# growth_rate = 48
# trans_1_filter = (stemfilter * 2) + (num_init_layer * growth_rate)
####################################################
###
###  num of  channels in the 1st conv,
###  num of layer in 1st conv
###  growth rate,
###  factor in transition layer)
###  num_class  ( class + 1)
###################################################
class DSOD(nn.HybridBlock):
    def __init__(self, stem_filter, num_init_layer, growth_rate, factor, num_class):
        super(DSOD, self).__init__()
        if factor == 0.5:
            self.factor = 2
        else:
            self.factor = 1
        self.num_cls = num_class
        self.sizes = [[.2, .2], [.37, .37], [.45, .45], [.54, .54], [.71, .71], [.88, .88]]  #
        self.ratios = [[1, 2, 0.5]] * 6
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        trans1_filter = ((stem_filter * 2) + (num_init_layer * growth_rate) // self.factor)

        self.backbone_fisrthalf = nn.HybridSequential()
        with self.backbone_fisrthalf.name_scope():
            self.backbone_fisrthalf.add(
                stemblock(stem_filter),
                DenseBlcok(6, growth_rate),
                transitionLayer(trans1_filter),
                DenseBlcok(8, growth_rate)

            )
        trans2_filter = ((trans1_filter) + (8 * growth_rate) // self.factor)
        trans3_filter = ((trans2_filter) + (8 * growth_rate) // self.factor)

        self.backbone_secondehalf = nn.HybridSequential()
        with self.backbone_secondehalf.name_scope():
            self.backbone_secondehalf.add(
                transitionLayer(trans2_filter),
                DenseBlcok(8, growth_rate),
                transitionLayer(trans3_filter, with_pool=False),
                DenseBlcok(8, growth_rate),
                transitionLayer(256, with_pool=False)
            )
        self.PC_layer = nn.HybridSequential()  # pool -> conv
        numPC_layer = [256, 256, 128, 128, 128]
        with self.PC_layer.name_scope():
            for i in range(5):
                self.PC_layer.add(
                    pool_conv(numPC_layer[i]),
                )
        self.CC_layer = nn.HybridSequential()  # conv1 -> conv3
        numCC_layer = [256, 128, 128, 128]
        with self.CC_layer.name_scope():
            for i in range(4):
                self.CC_layer.add(
                    conv_conv(numCC_layer[i])
                )

        self.class_predictors = nn.HybridSequential()
        with self.class_predictors.name_scope():
            for _ in range(6):
                self.class_predictors.add(
                    cls_predictor(self.num_anchors, self.num_cls)
                )

        self.box_predictors = nn.HybridSequential()
        with self.box_predictors.name_scope():
            for _ in range(6):
                self.box_predictors.add(
                    bbox_predictor(self.num_anchors)
                )

    def flatten_prediction(self, pred):
        return pred.transpose(axes=(0, 2, 3, 1)).flatten()

    def concat_predictions(self, preds):
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
        scale_2 = F.concat(out, PC_1, dim=1)

        anchors.append(MultiBoxPrior(
            scale_2, sizes=self.sizes[1], ratios=self.ratios[1]))
        class_preds.append(
            self.flatten_prediction(self.class_predictors[1](scale_2)))
        box_preds.append(
            self.flatten_prediction(self.box_predictors[1](scale_2)))

        scale_predict = scale_2
        for i in range(1, 5):
            PC_Predict = self.PC_layer[i](scale_predict)
            CC_Predict = self.CC_layer[i - 1](scale_predict)
            scale_predict = F.concat(PC_Predict, CC_Predict, dim=1)

            anchors.append(MultiBoxPrior(
                scale_predict, sizes=self.sizes[i + 1], ratios=self.ratios[i + 1]))
            class_preds.append(
                self.flatten_prediction(self.class_predictors[i + 1](scale_predict)))
            box_preds.append(
                self.flatten_prediction(self.box_predictors[i + 1](scale_predict)))

        # print(scale_predict.shape)

        anchors = self.concat_predictions(anchors)
        class_preds = self.concat_predictions(class_preds)
        box_preds = self.concat_predictions(box_preds)

        class_preds = class_preds.reshape(shape=(0, -1, self.num_cls + 1))

        return anchors, class_preds, box_preds


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)

        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pj = output.pick(label, axis=self._axis, keepdims=True)
        loss = - self._alpha * ((1 - pj) ** self._gamma) * pj.log()

        return loss.mean(axis=self._batch_axis, exclude=True)


class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return loss.mean(self._batch_axis, exclude=True)


######################################################
##
##
## parameter setting
##
##
#####################################################
data_shape = 512
batch_size = 4
rgb_mean = nd.array([123, 117, 104])
retrain = True
inference = True
inference_data = 'pikachu.jpg'
epoch = 0


def get_iterators(data_shape, batch_size):
    class_names = ['dummy', 'pikachu']
    num_class = len(class_names)
    train_iter = mx.image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec='train.rec',
        path_imgidx='train.idx',
        shuffle=True,
        mean=True
    )
    return train_iter, class_names


train_data, class_names = get_iterators(data_shape, batch_size)

train_data.reset()
batch = train_data.next()
img, labels = batch.data[0], batch.label[0]
print(img.shape)

train_data.reshape(label_shape=(3, 5))
ctx = mx.gpu()

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
        DSOD(64, 6, 48, 1, 1)  # 64 6 48 1 1
    )

box_loss = SmoothL1Loss()
cls_loss = FocalLoss()  # hard neg mining vs FocalLoss()
l1_loss = gluon.loss.L1Loss()
net.initialize()
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

cls_metric = metric.Accuracy()
box_metric = metric.MAE()

filename = 'DSOD.params'
if retrain:
    print('load last time weighting')
    net.load_params(filename, ctx=mx.gpu())

for epoch in range(1, epoch):
    train_data.reset()
    cls_metric.reset()
    box_metric.reset()
    tic = time.time()
    if epoch % 800 == 0:
        trainer.set_learning_rate(trainer.learning_rate * lr_decay)
    for i, batch in enumerate(train_data):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)

        with mx.autograd.record():
            anchors, class_preds, box_preds = net(x)
            box_target, box_mask, cls_target = training_targets(anchors, class_preds, y)

            loss1 = cls_loss(class_preds, cls_target)

            loss2 = l1_loss(box_preds, box_target, box_mask)

            loss = loss1 + 5 * loss2
        loss.backward()
        trainer.step(batch_size)

        cls_metric.update([cls_target], [class_preds.transpose((0, 2, 1))])
        box_metric.update([box_target], [box_preds * box_mask])

    print('Epoch %2d, train %s %.2f, %s %.5f, time %.1f sec' % (
        epoch, *cls_metric.get(), *box_metric.get(), time.time() - tic))

    net.save_params(filename)

net.save_params(filename)

if inference:

    def process_image(fname):
        with open(fname, 'rb') as f:
            im = image.imdecode(f.read())
        # resize to data_shape
        data = image.imresize(im, data_shape, data_shape)
        # minus rgb mean
        data = data.astype('float32') - rgb_mean
        # convert to batch x channel x height xwidth
        return data.transpose((2, 0, 1)).expand_dims(axis=0), im


    def predict(x):
        anchors, cls_preds, box_preds = net(x.as_in_context(ctx))

        cls_probs = nd.SoftmaxActivation(
            cls_preds.transpose((0, 2, 1)), mode='channel')

        return MultiBoxDetection(cls_probs, box_preds, anchors,
                                 force_suppress=True, clip=False, nms_threshold=0.1)  # ,nms_threshold=0.1


    def box_to_rect(box, color, linewidth=3):
        """convert an anchor box to a matplotlib rectangle"""
        box = box.asnumpy()
        return plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor=color, linewidth=linewidth)


    def display(im, out, threshold=0.5):
        tic = time.time()
        colors = ['blue', 'green', 'red', 'black', 'magenta']
        plt.imshow(im.asnumpy())
        for row in out:
            row = row.asnumpy()
            class_id, score = int(row[0]), row[1]
            if class_id < 0 or score < threshold:
                continue
            color = colors[class_id % len(colors)]
            box = row[2:6] * np.array([im.shape[0], im.shape[1]] * 2)
            rect = box_to_rect(nd.array(box), color, 2)
            plt.gca().add_patch(rect)

            text = class_names[class_id]
            plt.gca().text(box[0], box[1],
                           '{:s} {:.2f}'.format(text, score),
                           bbox=dict(facecolor=color, alpha=0.5),
                           fontsize=10, color='white')
        print(time.time() - tic)

        plt.show()


    x, im = process_image(inference_data)
    out = predict(x)
    display(im, out[0], threshold=0.6)





