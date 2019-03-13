import os
import time

import scipy
import scipy.misc
from mnist import MNIST
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import (
    FeedForwardNetwork, FullConnection, LinearLayer, RecurrentNetwork,
    SigmoidLayer
)
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.customxml.networkwriter import NetworkWriter

SIZE = 28 * 28
HIDDEN_SIZE = 25
OUTPUT_SIZE = 10

file_path = 'save.xml'


def save(net, file_path):
    NetworkWriter.writeToFile(net, file_path)


def load_or_create(file_path):
    if os.path.exists(file_path):
        net = NetworkReader.readFrom(file_path)
        print("Network loaded from: {}".format(file_path))
        return net
    else:
        net = RecurrentNetwork()

        inLayer = LinearLayer(SIZE)
        hiddenLayer = SigmoidLayer(HIDDEN_SIZE)
        outLayer = SoftmaxLayer(OUTPUT_SIZE)

        net.addInputModule(inLayer)
        net.addModule(hiddenLayer)
        net.addOutputModule(outLayer)

        theta1 = FullConnection(inLayer, hiddenLayer)
        theta2 = FullConnection(hiddenLayer, outLayer)

        net.addConnection(theta1)
        net.addConnection(theta2)

        net.sortModules()
        print("Network created")
        return net


mndata = MNIST('data_files')
images, labels = mndata.load_training()
net = load_or_create(file_path)


ds = SupervisedDataSet(SIZE, net.outdim)
for image, label in list(zip(images, labels)):
    values = scipy.zeros(ds.outdim)
    values[label] = 1
    ds.addSample(image, values)

train_ds, test_ds = ds.splitWithProportion(0.8)

trainer = BackpropTrainer(net, train_ds, learningrate=0.1, momentum=0.1)

print("Size of train_ds: {}".format(len(train_ds['target'])))


def get_test_data(train_count=1000):
    count = 0
    for i in range(0, train_count):
        result = net.activate(images[i])
        if result.argmax() == labels[i]:
            count += 1
    return train_count / float(count)


def test(start_number=0, columns=4, rows=5, fontsize=8):
    from matplotlib import pyplot as plt
    ax = []
    fig = plt.figure(figsize=(28, 28))

    for number, i in enumerate(range(start_number, start_number + columns * rows)):
        result = net.activate(images[i])
        print('Result: {}, value: {}, {}'.format(result.argmax(), labels[i], result.round(2)))

        data = [images[i][28 * n: 28 * (n + 1)] for n in range(0, 28)]

        # fig, ax = plt.subplots()
        ax.append(fig.add_subplot(rows, columns, number + 1))

        textstr = '\n'.join((
            'Result=%s' % result.argmax(),
            'Value=%s' % labels[i],
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax[-1].text(
            0.05, 0.95, textstr, transform=ax[-1].transAxes,
            fontsize=fontsize, verticalalignment='top', bbox=props
        )
        label = "index: {}".format(i)
        ax[-1].text(
            0.5, -0.1, label, size=fontsize, ha="center",
            transform=ax[-1].transAxes
        )

        plt.imshow(data, cmap='gray')
        plt.axis('off')
    plt.show()


def test_information(f):
    def new_f(*args, **kwargs):
        for totalepochs, exec_time, test_data in f(*args, **kwargs):
            print("".join((
                "epoch: %3d " % totalepochs,
                "\ttrain acc: %5.2f%% " % test_data,
                "\t{:.3f} seconds".format(exec_time),
            )))
    return new_f


@test_information
def train_epochs(epochs=100):
    for epoch in range(0, epochs):
        start_time = time.time()
        trainer.trainEpochs(1)
        save(net, file_path)
        yield trainer.totalepochs, time.time() - start_time, get_test_data()


@test_information
def train_until_convergence():
    start_time = time.time()
    train_mse, validation_mse = trainer.trainUntilConvergence(
        verbose=True,
        validationProportion=0.80,
        maxEpochs=100,
        continueEpochs=10,
        validationData=test_ds,
    )
    save(net, file_path)
    yield trainer.totalepochs, time.time() - start_time, get_test_data()


import pdb; pdb.set_trace()
