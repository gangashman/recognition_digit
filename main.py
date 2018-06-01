import os
import time

from mnist import MNIST
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError

HIDDEN_SIZE = 100
OUTPUT_SIZE = 9


def save(net, file_path):
    NetworkWriter.writeToFile(net, file_path)


def load_or_create(file_path):
    if os.path.exists(file_path):
        return NetworkReader.readFrom(file_path)
    else:
        net = buildNetwork(
            784, HIDDEN_SIZE, OUTPUT_SIZE,
            hiddenclass=TanhLayer,
            outclass=TanhLayer,
            bias=True
        )
        return net


mndata = MNIST('data_files')
images, labels = mndata.load_training()

file_path = 'save.xml'
net = load_or_create(file_path)


ds = SupervisedDataSet(784, OUTPUT_SIZE)
for image, label in list(zip(images, labels)):
    values = [0 for _ in range(0, 9)]
    values[label - 1] = 1
    ds.addSample(image, values)
trainer = BackpropTrainer(net, ds)


def test(number):
    result = net.activate(images[number])
    result = list(result).index(max(result))
    print('Result: {}, value: {}'.format(result, labels[number]))


def train_epochs(epochs):
    for epoch in range(0, epochs):
        start_time = time.time()
        trainer.trainEpochs(1)
        outTest = net.activateOnDataset(ds)
        outTest = outTest.argmax(axis=1)
        resTest = 100 - percentError(outTest, ds)

        print(
            "epoch: %4d " % trainer.totalepochs,
            "\ttest acc: %5.2f%%" % resTest,
            '\t{:.3f} seconds'.format(
                time.time() - start_time
            )
        )


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


train_epochs(1)
save(net, file_path)
test(1)
