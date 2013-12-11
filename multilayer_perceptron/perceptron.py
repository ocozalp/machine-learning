import numpy as np
from common.common_functions import sigmoid
import math


class BinaryPerceptron:

    def __init__(self, d):
        self.d = d
        self.weights = np.random.rand(d + 1)
        self.learning_rate = 0.01

    def threshold(self, value):
        if value >= 0.5:
            return 1
        return 0

    def train(self, input_set, result):
        for outer in xrange(10000):
            error = 0.0
            delta = np.zeros((self.d + 1,))

            for t in xrange(input_set.shape[0]):
                o = 0.0
                for i in xrange(self.d + 1):
                    o += self.weights.item((i,)) * input_set.item((t, i))

                y = sigmoid(o)
                response = self.threshold(y)

                if response != result.item((t,)):
                    error += 0.5 * (result.item((t,)) - o) * (result.item((t,)) - o)
                    for i in xrange(self.d + 1):
                        new_weight = (result.item((t,)) - y) * input_set.item((t, i))
                        delta.itemset((i,), delta.item((i,)) + new_weight)

            if error == 0.0:
                break

            for j in xrange(self.d + 1):
                self.weights.itemset((j,), self.weights.item((j,)) + self.learning_rate * delta.item((j,)))

    def response(self, input_set):
        o = 0.0
        for i in xrange(self.d + 1):
            o += self.weights.item((i,)) * input_set.item((i,))

        y = sigmoid(o)
        return self.threshold(y)


class MultiClassPerceptron:

    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.learning_rate = 0.01
        self.weights = np.random.rand(k, d + 1)

    def train(self, input_set, result):
        for outer in xrange(1000):
            error = 0.0
            delta = np.zeros((self.k, self.d + 1))

            for t in xrange(input_set.shape[0]):
                outputs = [0.0] * self.k
                positive_count = 0
                for i in xrange(self.k):
                    for j in xrange(self.d + 1):
                        outputs[i] += self.weights.item((i, j)) * input_set.item((t, j))

                    if outputs[i] > 0:
                        positive_count += 1

                y = [0.0] * self.k
                total = 0.0

                for i in xrange(self.k):
                    y[i] = math.pow(math.e, outputs[i])
                    total += y[i]

                m_index = 0
                for i in xrange(self.k):
                    y[i] /= total
                    if y[i] > y[m_index]:
                        m_index = i

                for i in xrange(self.k):
                    expected_result = result.item((t, i))
                    if (expected_result == 1 and outputs[i] <= 0.0) or (expected_result == 0 and outputs[i] >= 0.0):
                        error += 1
                        for j in xrange(self.d + 1):
                            delta.itemset((i, j), delta.item((i, j)) + (result.item((t, i)) - y[i]) * input_set.item((t, j)))

            if error == 0.0:
                break

            for i in xrange(self.k):
                for j in xrange(self.d + 1):
                    self.weights.itemset((i, j), self.weights.item((i, j)) + self.learning_rate * delta.item((i, j)))

    def response(self, input_set):
        outputs = [0.0] * self.k
        for i in xrange(self.k):
            for j in xrange(self.d + 1):
                outputs[i] += self.weights.item((i, j)) * input_set.item((j,))

        y = [0.0] * self.k
        total = 0.0

        for i in xrange(self.k):
            y[i] = math.pow(math.e, outputs[i])
            total += y[i]

        m_index = 0
        for i in xrange(self.k):
            y[i] /= total
            if y[i] > y[m_index]:
                m_index = i

        return m_index