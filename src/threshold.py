# aml - Machine learning library.
# Threshold methods module.
# Author: Julian Ailan
# ===================================

"""Provides a set of threshold methods."""

import numpy as np
import math as m

# Lambdas
def UnitStep(center=0):
    # return lambda x: 0 if x < center else (0.5 if x == center else 1)
    return lambda x: 0 if x < center else 1


def Logistic(center=0, width=1, derivative_needed=False):
    if derivative_needed:
        return (lambda x: m.exp((-x + center) / width) /
                pow((1 + m.exp((-x + center) / width)), 2))
    else:
        return lambda x: 1 / (1 + m.exp((-x + center) / width))


def TanH(center=0, width=1, derivative_needed=False):
    if derivative_needed:
        return lambda x: 1.0 - pow((x - center) / width, 2)
    else:
        return lambda x: np.tanh((x - center) / width)

# Non-lambdas
def logistic(data):
    """Performs the computation of an activation using the logistic function as
    sigmoid.

    Args:
        data: value to replace in the logistic function expression.

    Returns:
        The result of the operation.
    """
    return 1.0 / (1.0 + np.exp(-data))


def logistic_prime(data):
    """Performs the derivative of the logistic function, using its property.

    Args:
        data: value to replace in the logistic derivative function expression.

    Returns:
        The result of the operation.
    """
    return logistic(data) * (1 - logistic(data))


def tanh(x):
    """Performs the computation of an activation using the hyperbolic tangent
    function as sigmoid.

    Args:
        data: value to replace in the hyperbolic tangent function expression.

    Returns:
        The result of the operation.
    """
    return np.tanh(x)


def tanh_prime(x):
    """Performs the derivative of the hyperbolic tangent function.

    Args:
        data: value to replace in the hyperbolic tangent derivative function
        expression.

    Returns:
        The result of the operation.
    """
    return 1.0 - np.tanh(x) ** 2


activation_functions = {
    "logistic": logistic,
    "tanh": tanh,
}

activation_derivatives = {
    "logistic": logistic_prime,
    "tanh": tanh_prime,
}

lambdas = {
    "logistic": Logistic,
    "unitstep": UnitStep,
    "tanh": TanH
}
