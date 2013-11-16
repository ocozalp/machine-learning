"""
Commonly used statistics methods
"""
import numpy as np
import math
import common.linalg_wrapper as wrapper

def sample_mean(data):
    return sum(data) / len(data)


def sample_variance(data, mean=None):
    if mean is None:
        mean = sample_mean(data)

    return sum([(x - mean) ** 2 for x in data]) / len(data)


def get_covariance_matrix(sample_input):
    row, column = sample_input.shape
    means = np.zeros((column))

    for i in xrange(column):
        for j in xrange(row):
            means.itemset(i, means.item(i) + sample_input.item((j, i))/float(row))

    result = np.asmatrix(np.zeros((column, column)))

    for i in xrange(column):
        for j in xrange(i+1):
            total = 0.0
            for k in xrange(row):
                total += (sample_input.item((k, i)) - means.item(i))*(sample_input.item((k, j)) - means.item(j))

            total /= float(row - 1)
            result.itemset((i, j), total)
            result.itemset((j, i), total)
    return result


def mahalanobis_distance(x, mean, covariance_matrix):
    dif_vector = x - mean
    dif_vector = np.asmatrix(dif_vector)
    dif_vector_t = np.asmatrix(dif_vector).getT()

    return wrapper.dot_product_chain(dif_vector_t, covariance_matrix, dif_vector)


def multivariate_normal(x, mean, covariance_matrix):
    row, column = x.shape
    determinant_of_cov = np.asscalar(np.linalg.det(covariance_matrix))
    determinant_of_cov = math.sqrt(math.fabs(determinant_of_cov))

    result = (1.0/((math.pow((2.0 * math.pi), row/2.0))*determinant_of_cov))
    result *= math.pow(math.e, -0.5*mahalanobis_distance(x, mean, covariance_matrix))

    return result