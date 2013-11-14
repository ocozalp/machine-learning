"""
Linear regression example
"""
from random import random
from math import sqrt
import numpy as np


def get_result(x):
    w0 = sqrt(1.2)
    w1 = 1.1 ** 2

    return w1*x + w0


def get_noisy_result(x):
    res = get_result(x)
    if random() < 0.2:
        noise = random() * 0.05
        res += noise

    return res


def generate_sample_set(sample_size, func):
    inputs = [0.0] * sample_size
    outputs = [0.0] * sample_size

    for i in xrange(sample_size):
        inputs[i] = random()*10
        outputs[i] = func(inputs[i])

    return inputs, outputs


def get_y_vector(inputs, outputs):
    total_r = sum(outputs)
    total_r_x = sum([r*x for r, x in zip(outputs, inputs)])

    return np.matrix([[total_r], [total_r_x]])


def get_a_matrix(inputs):
    total_x = sum(inputs)
    total_x_sq = sum([x**2 for x in inputs])
    return np.matrix([[len(inputs), total_x], [total_x, total_x_sq]])

# w can be calculated using the formula w=A^-1 * y

n = 1000000
inputs, outputs = generate_sample_set(n, get_result)
y = get_y_vector(inputs, outputs)
a = get_a_matrix(inputs)
a_transpose = a.getI()
w = a_transpose.dot(y)
print w

#try for a new value

x = random()*10
print get_result(x)

input_vector = np.matrix([[1, x]])
new_output = input_vector.dot(w)

print new_output.item((0,0))