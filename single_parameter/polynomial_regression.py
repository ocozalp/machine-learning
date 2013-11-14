"""
Polynomial regression example
"""
from random import random
import numpy as np

degree = 4
coefficients = [0.5, 1.2, 2.4, 1.5, -0.2]


def get_result(x):
    result = 0
    num = 1
    for c in coefficients:
        result += c * num
        num *= x

    return result


def generate_sample_set(sample_size):
    inputs = [0.0] * sample_size
    outputs = [0.0] * sample_size

    for i in xrange(sample_size):
        inputs[i] = random()*10
        outputs[i] = get_result(inputs[i])

    return inputs, outputs


def get_y_vector(inputs, outputs):
    result = np.zeros((degree+1, 1))

    for input, output in zip(inputs, outputs):
        mul = output
        for i in xrange(degree + 1):
            result.itemset((i, 0), result.item((i, 0)) + mul)
            mul *= input

    return result


def get_a_matrix(inputs):
    result = np.zeros((degree + 1, degree + 1))
    result = np.asmatrix(result)

    for input in inputs:
        start = 1
        for i in xrange(degree + 1):
            mul = start
            for j in xrange(degree + 1):
                result.itemset((i, j), result.item((i, j)) + mul)
                mul *= input

            start *= input
    return result

n = 10000
input, output = generate_sample_set(n)
a = get_a_matrix(input)
a_inverse = a.getI()
y = get_y_vector(input, output)

w = a_inverse.dot(y)
print w


#try for a new value

x = random()*10
print get_result(x)

input_vector = np.ones((1, degree + 1))
x1 = x
for i in xrange(1, degree + 1):
    input_vector.itemset((0, i), x1)
    x1 *= x

new_output = input_vector.dot(w)

print new_output.item((0,0))