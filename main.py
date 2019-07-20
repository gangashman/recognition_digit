import os
import pickle

import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

mndata = MNIST('data_files')
images, labels = mndata.load_training()

file_path = 'save_file'

X = []
y = []
for image, label in list(zip(images, labels)):
    X.append(image)
    y.append(label)


X_train, X_test, y_train, y_test = train_test_split(
    np.array(X), np.array(y), test_size=0.5
)


def test(classifier, start_number=0, columns=4, rows=5, fontsize=8):
    from matplotlib import pyplot as plt
    ax = []
    fig = plt.figure(figsize=(28, 28))

    for number, i in enumerate(
            range(start_number, start_number + columns * rows)
    ):
        result = classifier.predict([images[i]])[0]

        print('Result: {}, value: {}'.format(result, labels[i]))

        data = [images[i][28 * n: 28 * (n + 1)] for n in range(0, 28)]

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


if os.path.exists(file_path):
    classifier = pickle.load(open(file_path, 'rb'))
else:
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    pickle.dump(classifier, open(file_path, '+wb'))

# test(classifier)
print(classifier.score(X_test, y_test))
