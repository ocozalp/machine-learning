import math


def sigmoid(val):
    result = math.e ** (-val)
    return 1.0 / (1 + result)