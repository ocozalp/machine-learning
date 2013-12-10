import random
import numpy as np
import math
import matplotlib.pyplot as plt


class LinearDiscriminator:

    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.learning_rate = 0.01
        self.weights = np.asmatrix(np.random.rand(k, d + 1) * 0.1)

    def train2(self, input_set, result):

        while True:
            delta = np.asmatrix(np.zeros((self.k, self.d+1)))
            err_count = 0

            for t in xrange(input_set.shape[0]):
                o = [0.0] * self.k
                y = [0.0] * self.k
                for i in xrange(self.k):
                    for j in xrange(self.d + 1):
                        o[i] += self.weights.item((i, j)) * input_set.item((t, j))

                s_o = sum([math.e ** oi for oi in o])
                m = 0
                m_val = -1000000000
                for i in xrange(self.k):
                    exp_res = (math.e ** o[i])
                    if exp_res > m_val:
                        m_val = exp_res
                        m = i
                    y[i] = (math.e ** o[i]) / s_o

                if result.item((t, m)) == 0:
                    err_count += 1

                for i in xrange(self.k):
                    for j in xrange(self.d + 1):
                        diff = (result.item((t, i)) - y[i]) * input_set.item((t, j))
                        delta.itemset((i, j), delta.item((i, j)) + diff)

            for i in xrange(self.k):
                for j in xrange(self.d + 1):
                    self.weights.itemset((i, j), self.weights.item((i, j)) + self.learning_rate * delta.item((i, j)))

            if err_count == 0:
                break

    def train(self, input_set, result):
        diverge = False
        last_error = input_set.shape[0]

        while not diverge:
            res = input_set * self.weights.getT()
            for i in xrange(res.shape[0]):
                s = 0.0
                for j in xrange(res.shape[1]):
                    s += math.e ** res.item((i, j))

                for j in xrange(res.shape[1]):
                    res.itemset((i, j), (math.e ** res.item((i, j))) / s)

            new_err = 0
            for i in xrange(res.shape[0]):
                max_index = 0
                m = res.item((i, 0))
                for j in xrange(res.shape[1]):
                    if res.item((i, j)) > m:
                        m = res.item((i, j))
                        max_index = j

                if result.item((i, max_index)) == 0:
                    new_err += 1

            if new_err > last_error:
                diverge = True
                print new_err
            else:
                res = (res - result).getT() * input_set
                self.weights -= self.learning_rate * res
                print new_err, last_error
                last_error = new_err

                if new_err <= 2:
                    break


def main():
    n = 15
    k = 3
    d = 2
    input_matrix = np.asmatrix(np.random.rand(n, d + 1))
    for i in xrange(n):
        input_matrix.itemset((i, 0), 1)

    for i in xrange(n/3):
        input_matrix.itemset((i, 0), random.randint(25, 50) / 100.0)
        input_matrix.itemset((i, 1), random.randint(25, 50) / 100.0)
        plt.plot([input_matrix.item((i, 0))], [input_matrix.item((i, 1))], 'bo')

    for i in xrange(n/3, n*2/3):
        input_matrix.itemset((i, 0), random.randint(125, 150) / 100.0)
        input_matrix.itemset((i, 1), random.randint(125, 150) / 100.0)
        plt.plot([input_matrix.item((i, 0))], [input_matrix.item((i, 1))], 'ro')

    for i in xrange(2*n/3, n):
        input_matrix.itemset((i, 0), random.randint(25, 50) / 100.0)
        input_matrix.itemset((i, 1), random.randint(125, 150) / 100.0)
        plt.plot([input_matrix.item((i, 0))], [input_matrix.item((i, 1))], 'go')

    result_vector = np.zeros((n, k))

    for i in xrange(n/3):
        result_vector.itemset((i, 0), 1)
    for i in xrange(n/3, 2*n/3):
        result_vector.itemset((i, 1), 1)
    for i in xrange(2*n/3, n):
        result_vector.itemset((i, 2), 1)

    ld = LinearDiscriminator(k, d)
    ld.train2(input_matrix, result_vector)

    for i in xrange(k):
        y1 = -(ld.weights.item((i, 0)) - 5 * ld.weights.item((i, 1))) / ld.weights.item((i, 2))
        y2 = -(ld.weights.item((i, 0)) + 5 * ld.weights.item((i, 1))) / ld.weights.item((i, 2))

        plt.plot([-5, 5], [y1, y2], 'g-')

    plt.show()

if __name__ == '__main__':
    main()