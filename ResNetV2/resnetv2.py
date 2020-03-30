################## Resnetv2
################## classification
################## minst 분류


################## 필요 함수
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
epochs=10
momentum=0.9
batch_size=64
#----------------#
# Hyperparameter #
##################

################## model
from mxnet.gluon.model_zoo import vision
net = vision.resnet18_v2(classes=10, pretrained=False, ctx=ctx)
# net = vision.resnet34_v2(classes=10, pretrained=False, ctx=ctx)
# net = vision.resnet50_v2(classes=10, pretrained=False, ctx=ctx)
# net = vision.resnet101_v2(classes=10, pretrained=False, ctx=ctx)
# net = vision.resnet152_v2(classes=10, pretrained=False, ctx=ctx)


################## 그래프
import gluoncv
inputShape = (1,3,224,224)
gluoncv.utils.viz.plot_network(net, shape=inputShape)


##### 전처리 ##############################################
def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32)
    return data, label

batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('dataset/MNIST', train = True, transform = transformer),
    batch_size = batch_size, shuffle = False, last_batch = 'discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('dataset/MNIST', train = False, transform = transformer),
    batch_size = batch_size, shuffle = True, last_batch = 'discard')

# 데이터 확인하기
for data, label in train_data:
    break

print(data.shape, label.shape)



### graph
from gluoncv.utils import viz
viz.plot_image(data[0][0])  # index 0 is image, 1 is label



### 최적화
net.collect_params().initialize(mx.init.Xavier(), ctx = ctx)

### trainer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'momentum': 0.9, 'learning_rate': .1})

# 오차 함수
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()

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
            loss = loss_function(output, label)
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
