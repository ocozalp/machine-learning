import numpy as np
import math
from common.common_functions import sigmoid


class LinearDiscriminator:

    def __init__(self, d):
        self.d = d
        self.learning_rate = 0.01
        self.weights = np.asmatrix(np.random.rand(d + 1) * 0.1)

    def train(self, input_set, result):
        last_error = 100000

        for outer in xrange(1000):
            error = 0.0
            delta = np.asmatrix(np.zeros((self.d + 1)))

            for t in xrange(input_set.shape[0]):
                o = 0
                for j in xrange(self.d + 1):
                    o += self.weights.item((j,)) * input_set.item((t, j))

                y = sigmoid(o)

                error += result.item((t,)) * math.log(y) + (1-result.item((t,))) * math.log(1-y)

                for j in xrange(self.d + 1):
                    delta.itemset((j,), delta.item((j,)) + (result.item((t,)) - y) * input_set.item((t, j)))

            error = -error
            if error > last_error:
                break

            last_error = error

            for j in xrange(self.d + 1):
                self.weights.itemset((j,), self.weights.item((j,)) + self.learning_rate * delta.item((j,)))