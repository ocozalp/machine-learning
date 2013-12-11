import random
import numpy as np
import matplotlib.pyplot as plt
from linear_disriminator import LinearDiscriminator


class MulticlassLinearDiscriminator:

    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.weights = np.asmatrix(np.zeros((k, d + 1)) * 0.1)

    def train(self, input_set, result):
        for ci in xrange(self.k):
            ld = LinearDiscriminator(self.d)
            new_result_set = np.asmatrix(np.zeros(input_set.shape[0]))

            for i in xrange(input_set.shape[0]):
                new_result_set.itemset((i,), result.item((i, ci)))

            ld.train(input_set, new_result_set)

            for i in xrange(self.d + 1):
                self.weights.itemset((ci, i), ld.weights.item((i,)))


def main():
    n = 15
    k = 3
    d = 2
    colors = ['b', 'r', 'g']
    input_matrix = np.asmatrix(np.zeros((n, d + 1)))

    for i in xrange(n):
        input_matrix.itemset((i, 0), 1)

    for i in xrange(n/3):
        input_matrix.itemset((i, 1), random.randint(25, 50) / 100.0)
        input_matrix.itemset((i, 2), random.randint(25, 50) / 100.0)
        plt.plot([input_matrix.item((i, 1))], [input_matrix.item((i, 2))], colors[0] + 'o')

    for i in xrange(n/3, (n*2)/3):
        input_matrix.itemset((i, 1), random.randint(125, 150) / 100.0)
        input_matrix.itemset((i, 2), random.randint(125, 150) / 100.0)
        plt.plot([input_matrix.item((i, 1))], [input_matrix.item((i, 2))], colors[1] + 'o')

    for i in xrange((2*n)/3, n):
        input_matrix.itemset((i, 1), random.randint(25, 50) / 100.0)
        input_matrix.itemset((i, 2), random.randint(125, 150) / 100.0)
        plt.plot([input_matrix.item((i, 1))], [input_matrix.item((i, 2))], colors[2] + 'o')

    result_vector = np.zeros((n, k))

    for i in xrange(n/3):
        result_vector.itemset((i, 0), 1)
    for i in xrange(n/3, (2*n)/3):
        result_vector.itemset((i, 1), 1)
    for i in xrange((2*n)/3, n):
        result_vector.itemset((i, 2), 1)

    ld = MulticlassLinearDiscriminator(k, d)
    ld.train(input_matrix, result_vector)

    for i in xrange(k):
        y1 = -(ld.weights.item((i, 0)) - 5 * ld.weights.item((i, 1))) / ld.weights.item((i, 2))
        y2 = -(ld.weights.item((i, 0)) + 5 * ld.weights.item((i, 1))) / ld.weights.item((i, 2))

        plt.plot([-5, 5], [y1, y2], colors[i] + '--')

    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.show()

if __name__ == '__main__':
    main()