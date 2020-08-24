# python
# -*- coding: utf-8 -*-
# Created by LiJing on 2020/5/26
from mxnet import image
import matplotlib.pyplot as plt
import numpy as np
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet import init
import common

dataRoot = 'D:\program\python\lijing\ImageSegmentation\data'
vocRoot = dataRoot + '/VOCdevkit/VOC2012'

classes = ['background','aeroplane','bicycle','bird','boat', 'bottle','bus','car','cat','chair','cow','diningtable', 'dog','horse','motorbike','person','potted plant', 'sheep','sofa','train','tv/monitor']
# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
[128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
[64,128,0],[192,128,0],[64,0,128],[192,0,128],
[64,128,128],[192,128,128],[0,64,0],[128,64,0],
[0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i

rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def rand_crop(data, label, height, width):
    data, rect = image.random_crop(data, (width, height))
    label = image.fixed_crop(label, *rect)
    return data, label

def ReadImage(root=vocRoot, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()
    n = len(images)
    data, label = [None] * n, [None] * n
    for i, fname in enumerate(images):
        data[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, fname))
        label[i] = image.imread('%s/SegmentationClass/%s.png' % (root, fname))
    return data, label

def show_images(images, nrows, ncols, figsize):
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            figs[i, j].imshow(images[i * ncols + j].asnumpy())
    plt.show()

def image2label(im):
    data = im.astype('int32').asnumpy()
    idx = (data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
    return nd.array(cm2lbl[idx])

def normalize_image(data):
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std

class VOCSegDataset(gluon.data.Dataset):
    def _filter(self, images):
            return [im for im in images if (
            im.shape[0] >= self.crop_size[0] and
            im.shape[1] >= self.crop_size[1])]

    def __init__(self, train, crop_size):
        self.crop_size = crop_size
        data, label = ReadImage(train=train)
        data = self._filter(data)
        self.data = [normalize_image(im) for im in data]
        self.label = self._filter(label)
        print('Read '+str(len(self.data))+' examples')

    def __getitem__(self, idx):
        data, label = rand_crop(
        self.data[idx], self.label[idx],
        *self.crop_size)
        data = data.transpose((2,0,1))
        label = image2label(label)
        return data, label

    def __len__(self):
        return len(self.data)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros(
        (in_channels, out_channels, kernel_size, kernel_size),
        dtype='float32')
    # for i in range(in_channels):
    #     if i < out_channels:
    #         j = i
    #     else:
    #         j = out_channels - 1
    #     weight[i, j, :, :] = filt
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)

def predict(im, net, ctx):
    data = normalize_image(im)
    data = data.transpose((2,0,1)).expand_dims(axis=0)
    yhat = net(data.as_in_context(ctx))
    pred = nd.argmax(yhat, axis=1)
    return pred.reshape((pred.shape[1], pred.shape[2]))

def label2image(pred):
    x = pred.astype('int32').asnumpy()
    cm = nd.array(colormap).astype('uint8')
    return nd.array(cm[x,:])

class CeShiUint(nn.HybridBlock):
    def __init__(self,channels,  **kwargs):
        super(CeShiUint, self).__init__(**kwargs)
        with self.name_scope():
            self.cov = nn.Conv2D(channels=channels, kernel_size=3, padding=1, strides=1)
            self.bn = nn.BatchNorm()

    def hybrid_forward(self, F, x):
        out = self.cov(x)
        out = self.bn(out)
        # return F.relu(out + x)
        return F.relu(out)

class CeShiNet(nn.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(CeShiNet, self).__init__(**kwargs)
        with self.name_scope():
            net = self.net = nn.HybridSequential()

            net.add(nn.Conv2D(channels=32, kernel_size=3, padding=1, strides=1, activation="relu"))
            for _ in range(1):
                net.add(CeShiUint(channels=32))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))

            net.add(nn.Conv2D(channels=64, kernel_size=3, padding=1, strides=1, activation="relu"))
            for _ in range(1):
                net.add(CeShiUint(channels=64))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))

            net.add(nn.Conv2D(channels=128, kernel_size=3, padding=1, strides=1, activation="relu"))
            for _ in range(2):
                net.add(CeShiUint(channels=128))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))

            net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, strides=1, activation="relu"))
            for _ in range(2):
                net.add(CeShiUint(channels=256))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))

            net.add(nn.Conv2D(channels=512, kernel_size=3, padding=1, strides=1, activation="relu"))
            for _ in range(2):
                net.add(CeShiUint(channels=512))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))

            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        return out


def GetNet(numClasses, ctx):
    net = CeShiNet(numClasses)
    net.initialize(ctx=ctx)
    net.hybridize()
    return net

def main():
    input_shape = (320, 480)
    voc_train = VOCSegDataset(True, input_shape)
    voc_test = VOCSegDataset(False, input_shape)

    batch_size = 4
    train_data = gluon.data.DataLoader(
        voc_train, batch_size, shuffle=True, last_batch='discard')
    test_data = gluon.data.DataLoader(
        voc_test, batch_size, last_batch='discard')

    # ctx = common.ChoiceGpu()
    ctx = common.ChoiceCpu()

    pretrained_net1 = models.resnet18_v2(pretrained=True)
    # pretrained_net2 = GetNet(10, ctx=ctx)
    # pretrained_net3 = models.vgg13(pretrained=True)
    # pretrained_net2.load_parameters("train.params")

    net = nn.HybridSequential()
    for layer in pretrained_net1.features[:-2]:
    # for layer in pretrained_net2.net[:-2]:
        net.add(layer)

    num_classes = len(classes)
    with net.name_scope():
        net.add(
            nn.Conv2D(num_classes, kernel_size=1),
            # nn.Conv2DTranspose(num_classes, kernel_size=4, padding=1, strides=2),
            # nn.Conv2DTranspose(num_classes, kernel_size=4, padding=1, strides=2),
            # nn.Conv2DTranspose(num_classes, kernel_size=4, padding=1, strides=2),
            # nn.Conv2DTranspose(num_classes, kernel_size=4, padding=1, strides=2),
            # nn.Conv2DTranspose(num_classes, kernel_size=4, padding=1, strides=2)
            # nn.Conv2DTranspose(num_classes, kernel_size=32, padding=8, strides=16)
            nn.Conv2DTranspose(channels=num_classes, kernel_size=64, padding=16, strides=32),
        )

    conv_trans1 = net[-1]
    conv_trans1.initialize(init=init.Zero())
    # conv_trans2 = net[-2]
    # conv_trans2.initialize(init=init.Zero())
    # conv_trans3 = net[-3]
    # conv_trans3.initialize(init=init.Zero())
    # conv_trans4 = net[-4]
    # conv_trans4.initialize(init=init.Zero())
    # conv_trans5 = net[-5]
    # conv_trans5.initialize(init=init.Zero())
    #
    net[-2].initialize(init=init.Xavier())
    x = nd.zeros((batch_size, 3, *input_shape))
    net(x)

    shape = conv_trans1.weight.data().shape
    conv_trans1.weight.set_data(bilinear_kernel(*shape[0:3]))
    conv_trans0 = net[1]
    print(conv_trans0.weight.data())
    # print(conv_trans0.weight.data())
    #
    # shape = conv_trans2.weight.data().shape
    # conv_trans2.weight.set_data(bilinear_kernel(*shape[0:3]))
    #
    # shape = conv_trans3.weight.data().shape
    # conv_trans3.weight.set_data(bilinear_kernel(*shape[0:3]))
    #
    # shape = conv_trans4.weight.data().shape
    # conv_trans4.weight.set_data(bilinear_kernel(*shape[0:3]))
    #
    # shape = conv_trans5.weight.data().shape
    # conv_trans5.weight.set_data(bilinear_kernel(*shape[0:3]))

    ctx = common.ChoiceGpu()
    net.collect_params().reset_ctx(ctx)

    net.load_parameters("train.params", ctx=ctx)

    loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)

    if True:
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': 1/batch_size, 'wd': 1e-3})

        common.Train(train_data, 10, net, loss,
                    trainer, batch_size, ctx)

    n = 6
    imgs = []
    data, label = ReadImage(train=False)

    for i in range(n):
        x = data[i]
        pred = label2image(predict(x, net, ctx))
        imgs += [x, pred, label[i]]

    show_images(imgs, nrows=n, ncols=3, figsize=(6, 10))
    print("ok")

if __name__ == '__main__':
    main()