import numpy as np
from numba import jit
from sklearn.neighbors import KDTree, BallTree
from collections import Counter
from mnist import MNIST
from typing import List
from statistics import mean

mndata = MNIST('data_files')
images, labels = mndata.load_training()

limit = 1000


def reshape(images, labels, limit=None):
    '''
    Resize images [28*28] and labels to
    list of images[28][28] and label
    '''
    return [(
        [image[28 * n: 28 * (n + 1)] for n in range(0, 28)],
        label
    ) for image, label in list(zip(images[:limit], labels[:limit]))]


def train_test_split(d, test_size: float):
    split_value = int(len(d) * test_size)
    return d[split_value:], d[:split_value]


data = reshape(images, labels, limit)
train, test = train_test_split(data, test_size=0.3)


# @jit(nopython=True)
def fit(train_data):
    max_value = 0
    fit_data = [[[] for _ in range(0, 28)] for _ in range(0, 28)]

    for data, r in train_data:
        for x, x_row in enumerate(data):
            for y, y_value in enumerate(x_row):

                fit_data[x][y].append((y_value, r))

                if max_value < y_value:
                    max_value = y_value

    return fit_data, max_value


def find_nearest(values, value, count, max_value, x=None, y=None):
    distanced_values = [(abs(v[0] - value), v[0], v[1]) for v in values]
    sorted_list = sorted(distanced_values, key=lambda d: d[0])

    if not sorted_list:
        return None

    neighbors = [result for distance, value, result in sorted_list[:count]]
    nearest_dist, nearest_ind = KDTree(np.array(values)).query(np.array(values), k=count)
    #if x == 10 and y == 10:
    #    breakpoint()
    return Counter(neighbors).most_common()[0][0]


def predict(data, fit_data, max_value, count=1, threshold=None):
    results = []
    for x in range(0, 28):
        for y in range(0, 28):
            value = data[x][y]

            if not threshold or value > threshold:
                near = find_nearest(
                    fit_data[x][y], value, count, max_value,
                    x, y
                )
                if near is None:
                    continue
                results.append(near)

    return int(mean(results))


class Classifier:
    _fit_data = None
    _max_value = None
    count = 5
    threshold = None

    def __init__(self, threshold=None, count=5):
        self.threshold = threshold
        self.count = count

    def predict(self, data_list: List):
        return [
            predict(
                d, self._fit_data, self._max_value,
                count=self.count, threshold=self.threshold
            )
            for d in data_list
        ]

    def fit(self, train_data):
        self._fit_data, self._max_value = fit(train_data)

    def score(self, data_list: List) -> float:
        images = [d[0] for d in data_list]
        labels = [d[1] for d in data_list]
        results = self.predict(images)

        print(results)
        print(labels)

        guess_right = [i for i, r in enumerate(results) if labels[i] == r]
        return len(guess_right) / len(data_list)

    def display(self, start_number=0, columns=4, rows=5, fontsize=8):
        from matplotlib import pyplot as plt
        ax = []
        fig = plt.figure(figsize=(28, 28))

        for number, i in enumerate(
                range(start_number, start_number + columns * rows)
        ):
            result = classifier.predict([images[i]])[0]

            print('Result: {}, value: {}'.format(result, labels[i]))

            # fig, ax = plt.subplots()
            ax.append(fig.add_subplot(rows, columns, number + 1))

            textstr = '\n'.join((
                'Result=%s' % result,
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


classifier = Classifier(threshold=100, count=10)
classifier.fit(train)

# classifier.display()
# print(classifier.predict([test[0][0]]))
score = classifier.score(test[0:20])
print(score)
