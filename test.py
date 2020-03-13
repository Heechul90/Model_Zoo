import sys
import mxnet as mx
import time
import numpy as np

from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn

##################
# Hyperparameter #
#----------------#
ctx = mx.cpu()
lr=0.05
epochs=2
momentum=0.9
batch_size=64
# kv=mx.kv.create("dist")
#----------------#
# Hyperparameter #
##################

from mxnet.gluon.model_zoo import vision
# net = vision.resnet34_v1()
# net = vision.resnet152_v2()
# net = vision.vgg11()
# net = vision.vgg11_bn()
# net = vision.alexnet()
# net = vision.densenet121()
# net = vision.squeezenet1_1()
# net = vision.inception_v3()
net = vision.mobilenet1_0()
# net = vision.mobilenet0_75()

############### 그래프 ###############
# import gluoncv
# inputShape = (1,3,224,224)
# gluoncv.utils.viz.plot_network(net, shape=inputShape)
#####################################

### 데이터
def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32)
    return data, label

train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR100('../dataset/cifar100', train=True, transform = transformer),
    batch_size = batch_size, shuffle = False, last_batch = 'discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR100('../dataset/cifar100', train=False, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')


# for d, l in train_data:
#     break
#
# print(d.shape, l.shape)


########################################################################################################################
### train
net.collect_params().initialize(mx.init.Xavier(magnitude = 0), ctx = ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .11})
# 오차 함수
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis = 1)
        acc.update(preds = predictions, labels = label)
    return acc.get()

#######
# Only one epoch so tests can run quickly, increase this variable to actually run
#######

epochs = 1
smoothing_constant = 0.01

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ############
        # keep a moving average of the losses
        ############

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

