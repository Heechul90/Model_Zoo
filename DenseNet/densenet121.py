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
kv=mx.kv.create("dist")
#----------------#
# Hyperparameter #
##################

from mxnet.gluon.model_zoo import vision
# net = vision.resnet152_v1()
# net = vision.resnet152_v2()
net = vision.vgg11(classes=100)
# net = vision.vgg11_bn()
# net = vision.alexnet()
net = vision.densenet121(classes=100)
# net = vision.squeezenet1_0()
# net = vision.inception_v3()
# net = vision.mobilenet1_0()
# net = vision.mobilenet_v2_0_25()

############### 그래프 ###############
import gluoncv
inputShape = (1,3,224,224)
gluoncv.utils.viz.plot_network(net, shape=inputShape)
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
def train():
    net.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(),
                            optimizer='sgd',
                            optimizer_params={'learning_rate':lr, 'momentum': momentum},
                            kvstore=kv)
#                            update_on_kvstore=True)

    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    stride=int(batch_size/kv.num_workers)
    start=kv.rank*stride
    end=start+stride
    if kv.rank==kv.num_workers:
       end=batch_size

    for epoch in range(epochs):
        metric.reset()
        for i, (data, label) in enumerate(train_data):
            if len(data)<start:
                trainer.step(batch_size)
                break
            elif len(data)<batch_size:
                end=len(data)

            data = data[start:end].as_in_context(ctx)
            label = label[start:end].as_in_context(ctx)
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            trainer.step(data.shape[0])
            metric.update([label], [output])
            print('[Epoch %d] Training'%(i))
            sys.stdout.flush()


        name, acc = metric.get()
        #if kv.rank==0:
        print('[Epoch %d] Training: %s=%f'%(epoch, name, acc))



# test
def test():
    metric = mx.metric.Accuracy()
    for data, label in test_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])
    name, acc = metric.get()
    print('Validation: %s=%f'%(name, acc))



if __name__ == '__main__':
    train()
#    if kv.rank==0:
#        test()

