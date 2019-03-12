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


SIZE = 28 * 28
HIDDEN_SIZE = 25
OUTPUT_SIZE = 10


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
        return net


mndata = MNIST('data_files')
images, labels = mndata.load_training()

file_path = 'save.xml'
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


def test(number):
    result = net.activate(images[number])
    print('Result: {}, value: {}'.format(result.argmax(), labels[number]))


def train_epochs(epochs):
    for epoch in range(0, epochs):
        start_time = time.time()
        trainer.trainEpochs(1)

        outTrain = net.activateOnDataset(train_ds)
        outTrain = outTrain.argmax(axis=1)
        resTrain = 100 - percentError(outTrain, true_train)

        outTest = net.activateOnDataset(test_ds)
        outTest = outTest.argmax(axis=1)
        resTest = 100 - percentError(outTest, true_test)

        save(net, file_path)
        print("".join((
            "epoch: %3d " % trainer.totalepochs,
            "\ttrain acc: %5.2f%% " % resTrain,
            "\ttest acc: %5.2f%%" % resTest,
            "\t{:.3f} seconds".format(time.time() - start_time),
        )))


def train_until_convergence():
    epochs = 100
    continue_epochs = 10
    validation_proportion = 0.15

    start_time = time.time()
    train_mse, validation_mse = trainer.trainUntilConvergence(
        verbose=True,
        validationProportion=validation_proportion,
        maxEpochs=epochs,
        continueEpochs=continue_epochs
    )
    print(
        'Training complete; {:.3f} seconds'.format(time.time() - start_time)
    )


import pdb; pdb.set_trace()
