# python
# -*- coding: utf-8 -*-
# Created by LiJing on 2020/5/18
import time
import os
import random
import shutil
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.data import vision
from mxnet.gluon.data.vision import transforms
import enum

NETMODE = enum.Enum("NETMODE", ("VGG", "RES", "TN"))

# 优先选择GPU
def ChoiceGpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1, 1), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

# 优先选择CPU
def ChoiceCpu():
    ctx = mx.cpu()
    return ctx

# 格式化成2016-03-20 11:45:39形式
def CurrentTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " "

# 计算精度
def Accuracy(outPut, label):
    return nd.mean(outPut.argmax(axis=1) == label).asscalar()

def EvaluateAccuracy(dataIterator, net):
    acc = 0.0
    for data, label in dataIterator:
        outLabel = net(data)
        acc += Accuracy(outLabel, label)
    return (acc/len(dataIterator))

# 重构图像数据集
def ReorgImageData(srcDataDir, dstDataDir, ratio=1):
    pathDir = os.listdir(srcDataDir)
    currentDataNum = len(pathDir)
    outDataNum = round(currentDataNum * ratio)
    sampleData = random.sample(pathDir, outDataNum)
    print(sampleData)
    for name in sampleData:
        shutil.copy(srcDataDir+"\\"+name, dstDataDir+"\\"+name)

# VGG模型
class VGGUint(nn.HybridBlock):
    def __init__(self, layers, channels, kernelSize, poolFlag=True, **kwargs):
        super(VGGUint, self).__init__(**kwargs)
        self.layers = layers
        self.poolFlag = poolFlag
        with self.name_scope():
            self.cov2 = nn.Conv2D(channels=channels, kernel_size=kernelSize, padding=1, strides=1, activation="relu")
            self.pool = nn.MaxPool2D(pool_size=2, strides=2)

    def hybrid_forward(self, F, x):
        out = x
        for _ in range(self.layers):
            out = self.cov2(out)
        if self.poolFlag:
            out = self.pool(out)
        return out

class VGGNet(nn.HybridBlock):
    def __init__(self, numClasses, **kwargs):
        super(VGGNet, self).__init__(**kwargs)
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            net.add(VGGUint(1, 64, 3))
            net.add(VGGUint(1, 128, 3))
            net.add(VGGUint(1, 256, 3, False))
            net.add(VGGUint(1, 256, 3))
            net.add(VGGUint(1, 512, 3, False))
            net.add(VGGUint(1, 512, 3))
            net.add(VGGUint(1, 512, 3, False))
            net.add(VGGUint(1, 512, 3))
            net.add(nn.Flatten())
            net.add(nn.Dense(4096))
            net.add(nn.Dense(4096))
            net.add(nn.Dense(numClasses))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        return out

# RES网络
class ResiUnit(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(ResiUnit, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.cov1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
            self.cov2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            if not same_shape:
                self.cov3 = nn.Conv2D(channels, kernel_size=1, strides=strides)
            self.bn1 = nn.BatchNorm()
            self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.cov1(x)))
        out = self.bn2(self.cov2(out))
        if not self.same_shape:
            x = self.cov3(x)
        return F.relu(out + x)

class ResiNet(nn.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(ResiNet, self).__init__(**kwargs)
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # 模块1
            net.add(nn.Conv2D(channels=32, kernel_size=3, padding=1, strides=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation="relu"))
            # 模块2
            for _ in range(3):
                net.add(ResiUnit(channels=32))
            # 模块 3
            net.add(ResiUnit(channels=64, same_shape=False))
            for _ in range(2):
                net.add(ResiUnit(channels=64))
            # 模块 4
            net.add(ResiUnit(channels=128, same_shape=False))
            for _ in range(2):
                net.add(ResiUnit(channels=128))
            # 模块 5
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        return out

class CeShiUint(nn.HybridBlock):
    def __init__(self,channels,  **kwargs):
        super(CeShiUint, self).__init__(**kwargs)
        with self.name_scope():
            self.cov = nn.Conv2D(channels=channels, kernel_size=3, padding=1, strides=1)

    def hybrid_forward(self, F, x):
        out = self.cov(x)
        return F.relu(out + x)

class CeShiNet(nn.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(CeShiNet, self).__init__(**kwargs)
        with self.name_scope():
            net = self.net = nn.HybridSequential()

            net.add(nn.Conv2D(channels=32, kernel_size=3, padding=1, strides=1, activation="relu"))
            for _ in range(2):
                net.add(CeShiUint(channels=32))
            # net.add(nn.Dropout(0.2))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))

            net.add(nn.Conv2D(channels=64, kernel_size=3, padding=1, strides=1, activation="relu"))
            for _ in range(2):
                net.add(CeShiUint(channels=64))
            # net.add(nn.Dropout(0.5))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))

            net.add(nn.Conv2D(channels=128, kernel_size=3, padding=1, strides=1, activation="relu"))
            for _ in range(2):
                net.add(CeShiUint(channels=128))
            # net.add(nn.Dropout(0.5))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))

            net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, strides=1, activation="relu"))
            for _ in range(2):
                net.add(CeShiUint(channels=256))
            # net.add(nn.Dropout(0.2))
            net.add(nn.MaxPool2D(pool_size=2, strides=2))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        return out

def GetNet(numClasses, netClass, ctx):
    if (NETMODE.VGG.value - 1) == netClass:
        net = VGGNet(numClasses)
    elif (NETMODE.RES.value - 1) == netClass:
        net = ResiNet(numClasses)
    elif (NETMODE.TN.value - 1) == netClass:
        net = CeShiNet(numClasses)

    net.initialize(ctx=ctx)
    net.hybridize()
    return net

def Train(train_data, iterations, net, softMaxLoss, trainer, batch_size, ctx):
    startDatatime = time.time()
    for epco in range(iterations):
        trainLoss = 0.0
        trainAcc = 0.0
        for data, label in train_data:
            data = data.as_in_context(ctx).astype("float32")
            label = label.as_in_context(ctx).astype("float32")
            with autograd.record():
                out = net(data)
                loss = softMaxLoss(out, label)
            loss.backward()
            trainer.step(batch_size)
            trainLoss += nd.mean(loss).asscalar()
            trainAcc += Accuracy(out, label)
        print("epco: %d, trainLoss: %f, trainAcc: %f" % (epco, trainLoss/len(train_data), trainAcc/len(train_data)))
    endDatatime = time.time()
    print(endDatatime - startDatatime)
    net.save_parameters("train.params")

# 定义损失函数
def LossFunction():
    softMaxLoss = gluon.loss.SoftmaxCrossEntropyLoss()
    return softMaxLoss

# 定义优化函数
def OptimizeFunction(net, lr):
    trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": lr, "wd": 0})
    return trainer

# 读取数据
def ReadData(batchSize):
    transform_train = transforms.Compose([
        # transforms.CenterCrop(32)
        # transforms.RandomFlipTopBottom(),
        # transforms.RandomColorJitter(brightness=0.0, contrast=0.0,saturation=0.0, hue=0.0),
        # transforms.RandomLighting(0.0),
        # transforms.Cast('float32'),
        # transforms.Resize(32),
        # 随机按照 scale 和 ratio 裁剪，并放缩为 32x32 的正⽅形
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        # 随机左右翻转图⽚
        transforms.RandomFlipLeftRight(),
        # 将图⽚像素值缩⼩到 (0,1) 内，并将数据格式从" ⾼ * 宽 * 通道" 改为" 通道 * ⾼ * 宽"
        transforms.ToTensor(),
        # 对图⽚的每个通道做标准化
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        # transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    data_dir = "D:\program\python\lijing\TrainUI"
    input_dir = "data"

    input_str = data_dir + '/' + input_dir + '/'

    train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1)
    valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1)
    # test_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1)
    test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1)

    loader = gluon.data.DataLoader

    trainData = loader(train_ds.transform_first(transform_train),
                        batchSize, shuffle=True, last_batch='keep')
    validData = loader(valid_ds.transform_first(transform_test),
                      batchSize, shuffle=True, last_batch='keep')
    testData = loader(test_ds.transform_first(transform_test),
                      batchSize, shuffle=False, last_batch='keep')

    return trainData, validData, testData

def main():
    a = 0
    print(a)

if __name__ == "__main__":
    main()