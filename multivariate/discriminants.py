import math
import numpy as np
import common.statistics as stats
import common.linalg_wrapper as wrapper


def simple_discriminant(covariance_matrix, mean_vector, input_vector, prior):
    row, column = mean_vector.shape
    determinant_of_cov = np.asscalar(np.linalg.det(covariance_matrix))

    result = -0.5*row*math.log(2*math.pi) - 0.5*math.log(determinant_of_cov) \
             - 0.5*stats.mahalanobis_distance(input_vector, mean_vector, covariance_matrix) + math.log(prior)

    return result


def quadratic_discriminant(covariance_matrix, mean_vector, input_vector, prior):
    determinant_of_cov = np.asscalar(np.linalg.det(covariance_matrix))
    input_transpose = np.asmatrix(input_vector).getT()
    inverse_covariance = covariance_matrix.getI()
    mean_transpose = np.asmatrix(mean_vector).getT()

    r1 = -0.5 * np.asscalar(wrapper.dot_product_chain(input_transpose, inverse_covariance, input_vector))

    wi = wrapper.dot_product_chain(inverse_covariance, mean_vector)
    wi = wi.getT()
    r2 = np.asscalar(wrapper.dot_product_chain(wi, input_vector))

    wi0 = -0.5*np.asscalar(wrapper.dot_product_chain(mean_transpose, inverse_covariance, mean_vector))
    wi0 += -0.5*math.log(determinant_of_cov) + math.log(prior)

    result = r1 + r2 + wi0

    return result


def common_covariance_discriminant(common_covariance_matrix, mean_vector, input_vector, prior):
    pass


def linear_discriminant(covariance_matrix, mean_vector, input_vector, prior):
    pass


