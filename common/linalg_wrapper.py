import numpy as np


def dot_product_chain(*args):
    result = None
    for arg in args:
        current_arg = arg
        current_type = type(current_arg)

        if current_type is not np.matrix:
            current_arg = np.asmatrix(current_arg)

        if result is None:
            result = current_arg
        else:
            result = result.dot(current_arg)

    return result