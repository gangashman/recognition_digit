import os
import time
import scipy

from mnist import MNIST
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.utilities import percentError
from pybrain.structure import FeedForwardNetwork, FullConnection
import scipy.misc


SIZE = 28 * 28
HIDDEN_SIZE = 3
OUTPUT_SIZE = 10

file_path = 'save.xml'


def save(net, file_path):
    NetworkWriter.writeToFile(net, file_path)


def load_or_create(file_path):
    if os.path.exists(file_path):
        return NetworkReader.readFrom(file_path)
    else:
        inLayer = LinearLayer(SIZE)
        hiddenLayer = SigmoidLayer(HIDDEN_SIZE)
        outLayer = SoftmaxLayer(OUTPUT_SIZE)

        net = FeedForwardNetwork()
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
true_train = train_ds['target'].argmax(axis=1)
true_test = test_ds['target'].argmax(axis=1)

trainer = BackpropTrainer(net, train_ds, learningrate=0.1, momentum=0.1)

print("Size of train_ds: {}".format(len(train_ds['target'])))


def get_test_data():
    outTrain = net.activateOnDataset(train_ds)
    outTrain = outTrain.argmax(axis=1)
    resTrain = 100 - percentError(outTrain, true_train)

    outTest = net.activateOnDataset(test_ds)
    outTest = outTest.argmax(axis=1)
    resTest = 100 - percentError(outTest, true_test)
    return resTrain, resTest


def test(number):
    from matplotlib import pyplot as plt
    result = net.activate(images[number])
    print('Result: {}, value: {}'.format(result.argmax(), labels[number]))

    data = [images[number][28 * i: 28 * (i + 1)] for i in range(0, 28)]

    fig, ax = plt.subplots()
    textstr = '\n'.join((
        'Result=%s' % result.argmax(),
        'Value=%s' % labels[number],
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    label = "Result max: {}".format(result.max())
    ax.text(0.5, -0.1, label, size=12, ha="center", transform=ax.transAxes)

    plt.imshow(data, interpolation='nearest')
    plt.show()


def test_information(f):
    def new_f(*args, **kwargs):
        for totalepochs, exec_time, test_data in f(*args, **kwargs):
            resTrain, resTest = test_data
            print("".join((
                "epoch: %3d " % totalepochs,
                "\ttrain acc: %5.2f%% " % resTrain,
                "\ttest acc: %5.2f%%" % resTest,
                "\t{:.3f} seconds".format(exec_time),
            )))
    return new_f


@test_information
def train_epochs(epochs):
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
