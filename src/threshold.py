# aml - Machine learning library.
# Threshold methods module.
# Author: Julian Ailan
# ===================================

"""Provides a set of threshold methods."""

import numpy as np


def logistic(data, center=0, width=1):
    """Performs the computation of an activation using the logistic function as
    sigmoid.

    Args:
        data: value to replace in the logistic function expression.

    Returns:
        The result of the operation.
    """
    return 1.0 / (1.0 + np.exp((-data + center) / width))


def logistic_prime(data, center=0, width=1):
    """Performs the derivative of the logistic function, using its property.

    Args:
        data: value to replace in the logistic derivative function expression.

    Returns:
        The result of the operation.
    """
    return logistic(data, center, width) * (1 - logistic(data, center, width))


def tan_h(data, center=0, width=1):
    """Performs the computation of an activation using the hyperbolic tangent
    function as sigmoid.

    Args:
        data: value to replace in the hyperbolic tangent function expression.

    Returns:
        The result of the operation.
    """
    return np.tanh((data - center) / width)


def tanh_prime(data, center=0, width=1):
    """Performs the derivative of the hyperbolic tangent function.

    Args:
        data: value to replace in the hyperbolic tangent derivative function
        expression.

    Returns:
        The result of the operation.
    """
    return 1.0 - tan_h(data, center, width) ** 2


def unit_step(data, center=0):
    """Simple unit-step function.

    Args:
        data: value along the step.

    Returns:
        The result of the operation.
    """
    return 0 if data < center else 1

activation_functions = {
    "logistic": logistic,
    "tanh": tan_h,
    "unitstep": unit_step
}

activation_derivatives = {
    "logistic": logistic_prime,
    "tanh": tanh_prime
}
