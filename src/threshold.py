# aml - Machine learning library.
# Threshold methods module.
# Author: Julian Ailan
# ===================================

"""Provides a set of threshold methods."""

# Third-party modules
import numpy as np


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
    output = logistic(data)
    return output * (1.0 - output)


def tan_h(data):
    """Performs the computation of an activation using the hyperbolic tangent
    function as sigmoid.

    Args:
        data: value to replace in the hyperbolic tangent function expression.

    Returns:
        The result of the operation.
    """
    return np.tanh(data)


def tan_h_prime(data):
    """Performs the derivative of the hyperbolic tangent function.

    Args:
        data: value to replace in the hyperbolic tangent derivative function
        expression.

    Returns:
        The result of the operation.
    """
    return 1.0 - np.tanh(data) ** 2


def unit_step(data, center=0):
    """Simple unit-step function.

    Args:
        data: value along the step.

    Returns:
        The result of the operation.
    """
    return 0 if data < center else 1


def relu(data, epsilon=0.1):
    """Leaky ReLU function.

    Args:
        data: value to replace in the max(epsilon, data) function.
        epsilon: step to generate some error to backpropagate and avoid neuron
        dying.

    Returns:
        The result of the Leaky ReLU operation.
    """
    return np.maximum(epsilon * data, data)


def relu_prime(data, epsilon=0.1):
    """Performs the derivative of the Leaky ReLU operation.

    Args:
        data: value to replace in the max(epsilon, data) function.
        epsilon: step to generate some error to backpropagate and avoid neuron
        dying.

    Returns:
        gradients: The result of the Leaky ReLU derivative operation.
    """
    gradients = 1. * (data > 0)
    gradients[gradients == 0] = epsilon
    return gradients


activation_functions = {
    "logistic": logistic,
    "tanh": tan_h,
    "unitstep": unit_step,
    "relu": relu
}

activation_derivatives = {
    "logistic": logistic_prime,
    "tanh": tan_h_prime,
    "relu": relu_prime
}
