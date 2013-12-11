import numpy as np
from linear_disriminator import LinearDiscriminator


class MultiClassLinearDiscriminator:

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